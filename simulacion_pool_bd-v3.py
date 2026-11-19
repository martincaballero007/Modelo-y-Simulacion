"""
Simulación de Eventos Discretos (Modelo G/G/c/K) de un Pool de Conexiones a Base de Datos — Versión 3
Implementación gemela en Python + SimPy sincronizada con index.html (v3) para GitHub Pages.

Características:
- Modela llegadas de Poisson (Exponencial), cola FIFO finita (K) con rechazos HTTP 503 y pool de c conexiones.
- Soporta distribuciones de servicio: Lognormal (CV=0.8), Normal truncada, Exponencial y Uniforme.
- Registra métricas por conexión individual (Conn #1 .. Conn #c), costo de reserva (conexión-segundos) y SLA.
- Valida el motor contra la fórmula analítica Erlang C (M/M/c) y la Ley de Little (L = lambda_ef * W).
- Ejecuta 30 réplicas independientes con Intervalo de Confianza al 95% (t de Student, 29 g.l.).

Requisitos:
    pip install simpy
"""

import math
import random
import statistics
import simpy

# ==============================================================================
# 1. PARÁMETROS GLOBALES Y ESCENARIOS EXPERIMENTALES
# ==============================================================================
SEMILLA_BASE = 42
T_SIM = 60.0          # Horizonte de simulación por réplica (segundos)
T_WARMUP = 6.0        # 10% inicial descartado para alcanzar estado estable
REPLICAS = 30         # Réplicas independientes por escenario
T_95 = 2.04523        # Valor crítico t de Student (95% confianza, 29 g.l.)
SLA_MS = 150.0        # Umbral SLA de respuesta total (ms)

ESCENARIOS = [
    ("E0: Base (As-Is)",          60.0, 5, 0.050, 40),
    ("E1: Sobrecarga (As-Is)",    92.0, 5, 0.050, 40),
    ("E2: Estrés Extremo",       115.0, 5, 0.050, 40),
    ("E3: Optimizado (To-Be)",    92.0, 8, 0.050, 40),
    ("E4: Pico Extremo (To-Be)", 115.0, 9, 0.050, 40),
]


# ==============================================================================
# 2. VALIDACIÓN ANALÍTICA: FÓRMULA DE ERLANG C (M/M/c)
# ==============================================================================
def erlang_c_wq_ms(lam, media_s, c):
    """Calcula el tiempo medio teórico en cola Wq (ms) para una cola M/M/c."""
    mu = 1.0 / media_s
    a = lam / mu
    rho = a / c
    if rho >= 1.0:
        return float("inf")
    suma = sum((a ** k) / math.factorial(k) for k in range(c))
    ultimo = (a ** c) / (math.factorial(c) * (1.0 - rho))
    p_espera = ultimo / (suma + ultimo)
    wq_seg = p_espera / (c * mu - lam)
    return wq_seg * 1000.0


def percentil(datos, p):
    if not datos:
        return 0.0
    xs = sorted(datos)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


# ==============================================================================
# 3. MOTOR DE SIMULACIÓN CON SIMPY (CON TRACKING INDIVIDUAL POR CONEXIÓN)
# ==============================================================================
def correr_replica(lam, c, media_s=0.050, k_max=40, dist="lognormal", t_sim=T_SIM, warmup=T_WARMUP, seed=42):
    rng_arr = random.Random(seed)
    rng_srv = random.Random(seed + 104729)

    env = simpy.Environment()
    pool = simpy.Resource(env, capacity=c)
    slots_libres = list(range(c))  # Identificadores Conn #1 .. Conn #c
    slot_tiempo_ocupado = [0.0] * c
    slot_atendidas = [0] * c
    activos = {}  # conexión -> instante de inicio del servicio en curso

    estado = {
        "t_prev": 0.0,
        "en_cola": 0,
        "en_srv": 0,
        "area_cola": 0.0,
        "area_srv": 0.0,
        "lq_max": 0,
        "llegadas": 0,
        "rechazos": 0,
    }

    wq_list = []
    w_list = []
    log_eventos = []

    def acumular():
        ahora = env.now
        ini = max(estado["t_prev"], warmup)
        if ahora > ini:
            dt = ahora - ini
            estado["area_cola"] += estado["en_cola"] * dt
            estado["area_srv"] += estado["en_srv"] * dt
        estado["t_prev"] = ahora

    def generar_servicio():
        if dist == "exponencial":
            return rng_srv.expovariate(1.0 / media_s)
        if dist == "normal":
            return max(0.005, rng_srv.gauss(media_s, media_s * 0.25))
        if dist == "uniforme":
            return rng_srv.uniform(0.5 * media_s, 1.5 * media_s)
        # Lognormal con CV = 0.8
        cv = 0.8
        sigma2 = math.log(1.0 + cv ** 2)
        sigma = math.sqrt(sigma2)
        mu = math.log(media_s) - 0.5 * sigma2
        return rng_srv.lognormvariate(mu, sigma)

    def proceso_consulta(case_id):
        t_llegada = env.now
        en_ventana = t_llegada >= warmup
        if en_ventana:
            estado["llegadas"] += 1

        # Control de admisión (búfer FIFO finito K)
        if estado["en_cola"] >= k_max:
            if en_ventana:
                estado["rechazos"] += 1
            return

        acumular()
        estado["en_cola"] += 1
        if en_ventana and estado["en_cola"] > estado["lq_max"]:
            estado["lq_max"] = estado["en_cola"]

        with pool.request() as req:
            yield req
            t_inicio = env.now
            acumular()
            estado["en_cola"] -= 1
            estado["en_srv"] += 1

            # Asignar la primera conexión física disponible
            slots_libres.sort()
            conn_idx = slots_libres.pop(0)
            activos[conn_idx] = t_inicio

            duracion = generar_servicio()
            yield env.timeout(duracion)

            t_fin = env.now
            acumular()
            estado["en_srv"] -= 1
            slots_libres.append(conn_idx)
            activos.pop(conn_idx, None)

            ini_ef = max(t_inicio, warmup)
            if t_fin > ini_ef:
                slot_tiempo_ocupado[conn_idx] += (t_fin - ini_ef)

            if en_ventana:
                slot_atendidas[conn_idx] += 1
                wq_ms = (t_inicio - t_llegada) * 1000.0
                w_ms = (t_fin - t_llegada) * 1000.0
                wq_list.append(wq_ms)
                w_list.append(w_ms)
                if len(log_eventos) < 7:
                    log_eventos.append((case_id, f"Conn #{conn_idx + 1}", t_llegada, t_inicio, t_fin, wq_ms, w_ms))

    def generador_llegadas():
        seq = 1
        while True:
            yield env.timeout(rng_arr.expovariate(lam))
            env.process(proceso_consulta(f"REQ-{seq:04d}"))
            seq += 1

    env.process(generador_llegadas())
    env.run(until=t_sim)
    acumular()
    # Cerrar el tiempo ocupado de las conexiones aún en servicio al final del horizonte
    for idx, t_ini in activos.items():
        ini_ef = max(t_ini, warmup)
        if t_sim > ini_ef:
            slot_tiempo_ocupado[idx] += (t_sim - ini_ef)

    t_med = t_sim - warmup
    W_ms = statistics.mean(w_list)
    Wq_ms = statistics.mean(wq_list)
    rho_pct = (estado["area_srv"] / (c * t_med)) * 100.0
    Lq = estado["area_cola"] / t_med
    L = (estado["area_cola"] + estado["area_srv"]) / t_med
    admitidas = max(0, estado["llegadas"] - estado["rechazos"])
    lam_ef = admitidas / t_med
    L_little = lam_ef * (W_ms / 1000.0)
    throughput = len(w_list) / t_med
    rechazo_pct = (estado["rechazos"] / estado["llegadas"]) * 100.0 if estado["llegadas"] > 0 else 0.0
    fuera_sla_pct = (sum(1 for x in w_list if x > SLA_MS) / len(w_list)) * 100.0
    costo_conn_s = c * t_med

    per_conn = [
        {"conn": f"Conn #{i + 1}", "atendidas": slot_atendidas[i], "util_pct": (slot_tiempo_ocupado[i] / t_med) * 100.0}
        for i in range(c)
    ]

    return {
        "Wq_ms": Wq_ms,
        "W_ms": W_ms,
        "P95_ms": percentil(w_list, 95),
        "rho_pct": rho_pct,
        "Lq": Lq,
        "L": L,
        "L_little": L_little,
        "throughput": throughput,
        "rechazo_pct": rechazo_pct,
        "fuera_sla_pct": fuera_sla_pct,
        "costo_conn_s": costo_conn_s,
        "per_conn": per_conn,
        "log": log_eventos,
    }


def resumir_30_replicas(resultados):
    resumen = {}
    claves = ["Wq_ms", "W_ms", "P95_ms", "rho_pct", "Lq", "L", "L_little", "throughput", "rechazo_pct", "fuera_sla_pct", "costo_conn_s"]
    for clave in claves:
        vals = [r[clave] for r in resultados]
        m = statistics.mean(vals)
        hw = T_95 * statistics.stdev(vals) / math.sqrt(len(vals))
        resumen[clave] = (m, hw)
    return resumen


# ==============================================================================
# 4. EJECUCIÓN PRINCIPAL: VALIDACIÓN ERLANG C + 30 RÉPLICAS POR ESCENARIO
# ==============================================================================
if __name__ == "__main__":
    print("=" * 94)
    print(" 1. VALIDACIÓN ANALÍTICA DEL MOTOR SIMPY CONTRA ERLANG C (M/M/c, K=infinito)")
    print("=" * 94)
    for c_val in (5, 8):
        reps_mm = [
            correr_replica(lam=60.0, c=c_val, media_s=0.050, k_max=10**9, dist="exponencial", seed=SEMILLA_BASE + r)
            for r in range(REPLICAS)
        ]
        s_mm = resumir_30_replicas(reps_mm)
        wq_teo = erlang_c_wq_ms(60.0, 0.050, c_val)
        wq_sim, wq_hw = s_mm["Wq_ms"]
        print(f" M/M/{c_val} (λ=60, μ=20): Wq simulado = {wq_sim:6.2f} ± {wq_hw:4.2f} ms | Wq teórico Erlang C = {wq_teo:6.2f} ms")

    print("\n" + "=" * 94)
    print(" 2. EXPERIMENTACIÓN DE ESCENARIOS (30 RÉPLICAS · IC 95 % · SERVICIO LOGNORMAL)")
    print("=" * 94)
    header = (
        f"{'Escenario':<25} | {'Wq (ms)':<14} | {'W/CT (ms)':<14} | {'P95 (ms)':<13} | "
        f"{'ρ (%)':<12} | {'Lq':<11} | {'L (WIP)':<11} | {'Rechazo':<8} | {'Fuera SLA':<9}"
    )
    print(header)
    print("-" * len(header))

    for nombre, lam, c, media_s, k_max in ESCENARIOS:
        reps = [
            correr_replica(lam=lam, c=c, media_s=media_s, k_max=k_max, dist="lognormal", seed=SEMILLA_BASE + r)
            for r in range(REPLICAS)
        ]
        s = resumir_30_replicas(reps)
        print(
            f"{nombre:<25} | "
            f"{s['Wq_ms'][0]:6.2f}±{s['Wq_ms'][1]:5.2f} | "
            f"{s['W_ms'][0]:6.2f}±{s['W_ms'][1]:5.2f} | "
            f"{s['P95_ms'][0]:6.1f}±{s['P95_ms'][1]:4.1f} | "
            f"{s['rho_pct'][0]:5.1f}±{s['rho_pct'][1]:4.1f}% | "
            f"{s['Lq'][0]:5.2f}±{s['Lq'][1]:4.2f} | "
            f"{s['L'][0]:5.2f}±{s['L'][1]:4.2f} | "
            f"{s['rechazo_pct'][0]:6.2f}% | "
            f"{s['fuera_sla_pct'][0]:7.1f}%"
        )

    print("\n--- DESGLOSE POR CONEXIÓN INDIVIDUAL (ESCENARIO E3: λ=92, c=8, RÉPLICA 0) ---")
    r_e3 = correr_replica(lam=92.0, c=8, media_s=0.050, k_max=40, dist="lognormal", seed=SEMILLA_BASE)
    for item in r_e3["per_conn"]:
        print(f" {item['conn']:<8} -> Atendidas: {item['atendidas']:4d} consultas | Utilización: {item['util_pct']:5.1f} %")

    print("\n--- MUESTRA DEL LOG DE EVENTOS (ESCENARIO E0, RÉPLICA 0) ---")
    r0 = correr_replica(lam=60.0, c=5, media_s=0.050, k_max=40, dist="lognormal", seed=SEMILLA_BASE)
    for cid, conn_name, t_ll, t_ini, t_fin, wq_ms, w_ms in r0["log"]:
        print(f" {cid} [{conn_name}] | Llegada={t_ll:7.4f}s | Inicio={t_ini:7.4f}s | Fin={t_fin:7.4f}s | Wq={wq_ms:6.2f}ms | W={w_ms:6.2f}ms")
