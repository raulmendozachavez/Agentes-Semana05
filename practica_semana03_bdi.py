"""
Práctica de Laboratorio - Semana 03: Agente deliberativo BDI
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

# Constantes de diseño (se definen fuera de las funciones para poder ajustarlas)
UMBRAL_CPU_CRITICO = 80.0     # CPU crítica (%)
UMBRAL_TASA_FUGA = 15.0       # incremento mínimo de CPU por ciclo (puntos %)
UMBRAL_LATENCIA_MS = 500.0    # latencia a partir de la cual el SLA está en riesgo


@dataclass
class TelemetriaCluster:
    ciclo: int
    cpu_pct: float
    latencia_ms: float
    alerta_siem: bool
    cuota_agotada: bool = False


@dataclass
class Deseo:
    nombre: str
    prioridad: int


@dataclass
class PlanBDI:
    nombre: str
    intencion_meta: str
    precondicion: Callable[[Dict[str, Any]], bool]
    cuerpo_acciones: List[str]


class AgenteBDI_Estudiante:
    def __init__(self):
        self.creencias: Dict[str, Any] = {
            "historial_cpu": [],
            "anomalia_fuga_memoria": False,
            "bajo_ataque": False,
            "cuota_cloud_agotada": False,
            "plan_fallido": None,
            # Creencias auxiliares que se alimentan de la telemetría
            "cpu_actual": 0.0,
            "latencia_ms": 0.0,
        }
        self.intencion_actual: Optional[Deseo] = None
        self.plan_activo: Optional[PlanBDI] = None
        self.cola_acciones: List[str] = []

    # TAREA 1: brf (revisión de creencias) con inferencia deductiva
    def brf(self, p: TelemetriaCluster) -> None:
        """Actualiza creencias y deriva 'anomalia_fuga_memoria'."""
        # Ventana deslizante: solo se conservan las últimas 3 lecturas
        self.creencias["historial_cpu"].append(p.cpu_pct)
        if len(self.creencias["historial_cpu"]) > 3:
            self.creencias["historial_cpu"].pop(0)

        # Creencias directas (percepción -> creencia)
        self.creencias["cpu_actual"] = p.cpu_pct
        self.creencias["latencia_ms"] = p.latencia_ms
        self.creencias["bajo_ataque"] = p.alerta_siem
        self.creencias["cuota_cloud_agotada"] = p.cuota_agotada

        # Regla de inferencia deductiva:
        #   Si p1 < p2 < p3 (estrictamente creciente) y la tasa media de
        #   incremento por ciclo >= 15  =>  anomalia_fuga_memoria = True
        # Se evalúa SIN esperar a que la CPU llegue al 80 %.
        h = self.creencias["historial_cpu"]
        if len(h) == 3:
            p1, p2, p3 = h
            estrictamente_creciente = p1 < p2 < p3
            tasa_por_ciclo = (p3 - p1) / 2.0   # 2 intervalos entre 3 lecturas
            self.creencias["anomalia_fuga_memoria"] = (
                estrictamente_creciente and tasa_por_ciclo >= UMBRAL_TASA_FUGA
            )
        else:
            # Con menos de 3 lecturas no hay evidencia suficiente
            self.creencias["anomalia_fuga_memoria"] = False

    # TAREA 2: options (generación de deseos)
    def options(self) -> Dict[str, Deseo]:
        """Genera el conjunto de deseos D según las creencias actuales."""
        c = self.creencias
        opciones: Dict[str, Deseo] = {}

        if c["bajo_ataque"]:
            opciones["MITIGAR_INTRUSION_SEGURIDAD"] = Deseo("MITIGAR_INTRUSION_SEGURIDAD", 100)

        # Supuesto: CPU crítica indica degradación del servicio de caché
        if c["cpu_actual"] >= UMBRAL_CPU_CRITICO:
            opciones["RESTAURAR_SERVICIO_CACHE"] = Deseo("RESTAURAR_SERVICIO_CACHE", 80)

        if c["anomalia_fuga_memoria"]:
            opciones["PREVENIR_CAIDA_OOM"] = Deseo("PREVENIR_CAIDA_OOM", 75)

        if c["latencia_ms"] > UMBRAL_LATENCIA_MS:
            opciones["ESTABILIZAR_RENDIMIENTO_SLA"] = Deseo("ESTABILIZAR_RENDIMIENTO_SLA", 60)

        return opciones

    # TAREA 3: filter (selección de la intención)
    def filter(self, opciones: Dict[str, Deseo]) -> Optional[Deseo]:
        """Compromiso de mente abierta: se reevalúa en cada ciclo y se elige
        el deseo de mayor prioridad."""
        if not opciones:
            return None
        return max(opciones.values(), key=lambda d: d.prioridad)

    # TAREA 4: selección de plan con soporte de Fallback
    def seleccionar_plan_fallback(self, intencion: Deseo,
                                  biblioteca: List[PlanBDI]) -> Optional[PlanBDI]:
        """Devuelve el primer plan de la biblioteca que sirva para la
        intención, cuya precondición se cumpla y que no esté marcado como fallido.
        El orden de la biblioteca define la jerarquía: principal antes que fallback."""
        for plan in biblioteca:
            if plan.intencion_meta != intencion.nombre:
                continue
            if plan.nombre == self.creencias["plan_fallido"]:
                continue
            if plan.precondicion(self.creencias):
                return plan
        return None

    # Ciclo de control completo del agente
    def ejecutar_ciclo(self, p: TelemetriaCluster, biblioteca: List[PlanBDI]) -> None:
        self.brf(p)
        nueva = self.filter(self.options())

        print(f"\n[Ciclo {p.ciclo}] CPU={p.cpu_pct}% lat={p.latencia_ms}ms "
              f"SIEM={p.alerta_siem} cuota_agotada={p.cuota_agotada}")
        print(f"  Creencias: historial={self.creencias['historial_cpu']} "
              f"fuga={self.creencias['anomalia_fuga_memoria']}")

        if nueva is None:
            self.intencion_actual, self.plan_activo, self.cola_acciones = None, None, []
            print("  Sin deseos activos: el agente no hace nada.")
            return

        # Reconsideración: si cambia la intención, se descarta el plan anterior
        if self.intencion_actual is None or nueva.nombre != self.intencion_actual.nombre:
            self.intencion_actual = nueva
            self.plan_activo = None
            self.cola_acciones = []
        print(f"  Intención: {self.intencion_actual.nombre} (prioridad {self.intencion_actual.prioridad})")

        # Si la cuota vuelve a estar disponible, se rehabilita el plan marcado
        if not self.creencias["cuota_cloud_agotada"]:
            self.creencias["plan_fallido"] = None

        # Detección de plan inviable en plena ejecución
        if self.plan_activo is not None and not self.plan_activo.precondicion(self.creencias):
            print(f"  >> Plan '{self.plan_activo.nombre}' INVIABLE. Se mantiene la intención.")
            self.creencias["plan_fallido"] = self.plan_activo.nombre
            self.plan_activo = None
            self.cola_acciones = []

        # Selección de plan (principal o fallback)
        if self.plan_activo is None:
            plan = self.seleccionar_plan_fallback(self.intencion_actual, biblioteca)
            if plan is None:
                print("  No hay plan aplicable para la intención.")
                return
            self.plan_activo = plan
            self.cola_acciones = list(plan.cuerpo_acciones)
            print(f"  Plan seleccionado: {plan.nombre}")

        # Ejecución de UNA acción por ciclo
        accion = self.cola_acciones.pop(0)
        print(f"  Acción ejecutada: {accion}")
        if not self.cola_acciones:
            print(f"  Plan '{self.plan_activo.nombre}' completado.")
            self.plan_activo = None


def construir_biblioteca() -> List[PlanBDI]:
    """Biblioteca de planes Π. El orden importa: el plan principal va primero."""
    return [
        PlanBDI("Runbook_Bloqueo_WAF", "MITIGAR_INTRUSION_SEGURIDAD",
                lambda b: True, ["BLOQUEAR_IPS_WAF", "AISLAR_SEGMENTO_RED"]),
        PlanBDI("Runbook_Reinicio_Cache", "RESTAURAR_SERVICIO_CACHE",
                lambda b: True, ["REINICIAR_NODO_CACHE", "PRECALENTAR_CACHE"]),
        PlanBDI("Runbook_Reinicio_Controlado_Pods", "PREVENIR_CAIDA_OOM",
                lambda b: True, ["DRENAR_TRAFICO_POD", "REINICIAR_POD_CON_FUGA"]),
        # Plan principal: requiere cuota de nube disponible
        PlanBDI("Runbook_Autoescalado_Horizontal", "ESTABILIZAR_RENDIMIENTO_SLA",
                lambda b: not b["cuota_cloud_agotada"],
                ["APROVISIONAR_NODO_REPLICA_K8S", "REBALANCEAR_TRAFICO_INGRESS"]),
        # Plan secundario (fallback): no depende de la cuota
        PlanBDI("Runbook_Degradacion_Elegante_CircuitBreaker", "ESTABILIZAR_RENDIMIENTO_SLA",
                lambda b: True,
                ["ACTIVAR_CIRCUIT_BREAKER_RECOMENDACIONES", "HABILITAR_RESPUESTA_ESTATICA_CDN"]),
    ]


if __name__ == "__main__":
    print("=== EVALUACIÓN EJERCICIO 1: AGENTE BDI ===")
    biblioteca = construir_biblioteca()

    print("\n##### ESCENARIO A: fuga de memoria (CPU 40 -> 58 -> 75) #####")
    agente = AgenteBDI_Estudiante()
    for i, cpu in enumerate([40.0, 58.0, 75.0], start=1):
        agente.ejecutar_ciclo(TelemetriaCluster(i, cpu, 200.0, False), biblioteca)

    print("\n##### ESCENARIO B: fuga de memoria + ciberataque (prioridad 100) #####")
    agente = AgenteBDI_Estudiante()
    lecturas = [(40.0, False), (58.0, False), (75.0, True)]
    for i, (cpu, siem) in enumerate(lecturas, start=1):
        agente.ejecutar_ciclo(TelemetriaCluster(i, cpu, 200.0, siem), biblioteca)

    print("\n##### ESCENARIO C: fallback por cuota cloud agotada #####")
    agente = AgenteBDI_Estudiante()
    agente.ejecutar_ciclo(TelemetriaCluster(1, 50.0, 900.0, False, False), biblioteca)
    agente.ejecutar_ciclo(TelemetriaCluster(2, 50.0, 900.0, False, True), biblioteca)
    agente.ejecutar_ciclo(TelemetriaCluster(3, 50.0, 900.0, False, True), biblioteca)