# Documentación del Sistema de Orquestación de Pipeline

## Índice
1. [Visión General](#visión-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Componentes Principales](#componentes-principales)
4. [Flujo de Ejecución](#flujo-de-ejecución)
5. [Diagrama de Conexiones](#diagrama-de-conexiones)
6. [Guía de Mantenimiento](#guía-de-mantenimiento)
7. [Cómo Agregar una Nueva Stage](#cómo-agregar-una-nueva-stage)
8. [Cómo Modificar Componentes Existentes](#cómo-modificar-componentes-existentes)
9. [Resolución de Problemas](#resolución-de-problemas)

---

## Visión General

El sistema de orquestación es responsable de coordinar la ejecución secuencial de múltiples etapas (stages) de un pipeline de Machine Learning. 

**Objetivos principales:**
- ✓ Validar precondiciones antes de ejecutar cada stage
- ✓ Manejar errores temporales con reintentos automáticos
- ✓ Rastrear estado y métricas de cada ejecución
- ✓ Proporcionar visibilidad completa del proceso
- ✓ Permitir agregar nuevas stages sin modificar código existente

---

## Arquitectura del Sistema

### Capas de la Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                      main.py (Entrada)                      │
├─────────────────────────────────────────────────────────────┤
│                        Orchestrator                         │
│  (Orquestador Principal - Coordinador del Pipeline)        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Adapters   │  │  Validators  │  │   Execution │      │
│  │ (Bridges)    │  │  (Guards)    │  │   Tracking  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Pipeline Infrastructure Layer                       │  │
│  │  - PipelineStage (Interface)                        │  │
│  │  - StageValidator (Interface)                       │  │
│  │  - StageExecution (Data Model)                      │  │
│  │  - ExecutionStatus (Enum)                           │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Existing Stages (Sin Cambios)                      │  │
│  │  - DataInspectionStage                              │  │
│  │  - FeatureSelectionStage                            │  │
│  │  - ModelSelectionStage                              │  │
│  │  - FineTuningStage                                  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Componentes Principales

### 1. Orchestrator (Orquestador)
**Ubicación:** `app/core/orchestrator/orchestrator.py`

**Responsabilidades:**
- Construir el pipeline de stages a ejecutar
- Ejecutar cada stage con validación previa
- Implementar lógica de reintentos automáticos
- Ejecutar la fase de evaluación después de cada stage
- Rastrear métricas de ejecución (timing, estado, errores)
- Generar reportes detallados

**Método Principal:**
```python
orchestrator = Orchestrator(context, max_retries=2)
summary = orchestrator.run()  # Retorna: {'success': [...], 'failed': [...], 'skipped': [...]}
```

---

### 2. PipelineStage (Interfaz de Stage)
**Ubicación:** `app/core/orchestrator/pipeline_stage.py`

**Interfaz que define el contrato para cada stage:**

```python
class PipelineStage(ABC):
    @abstractmethod
    def get_stage_type(self) -> Stages:
        """Retorna el tipo de stage (DATA_HANDLER, FEATURE_SELECTION, etc.)"""
        pass
    
    @abstractmethod
    def execute(self, context: RunContext) -> None:
        """Ejecuta la lógica del stage"""
        pass
    
    @abstractmethod
    def get_validator(self) -> Optional[StageValidator]:
        """Retorna el validador de precondiciones"""
        pass
```

**Responsabilidades de cualquier clase que implemente PipelineStage:**
- Identificarse a sí misma mediante `get_stage_type()`
- Ejecutar su lógica mediante `execute()`
- Proporcionar validación mediante `get_validator()`

---

### 3. StageValidator (Validador de Precondiciones)
**Ubicación:** `app/core/orchestrator/stage_validator.py` y `validators.py`

**Responsabilidades:**
- Verificar que las precondiciones de una stage se cumplan
- Retornar un resultado booleano con razón del fallo si es necesario

**Ejemplo: FeatureSelectionValidator**
```python
class FeatureSelectionValidator(StageValidator):
    def validate(self, context: RunContext) -> Tuple[bool, Optional[str]]:
        # ¿Existe DATA_HANDLER?
        if Stages.DATA_HANDLER not in context.stage_results:
            return False, "DATA_HANDLER no completado"
        
        # ¿Tiene resultados?
        if not context.stage_results[Stages.DATA_HANDLER].results:
            return False, "DATA_HANDLER no produjo resultados"
        
        return True, None
```

---

### 4. Adapters (Adaptadores de Stages)
**Ubicación:** `app/core/orchestrator/stages_adapters.py`

**Responsabilidades:**
- Adaptar las stages existentes a la interfaz PipelineStage
- Mantener compatibilidad hacia atrás con código existente
- Conectar validadores con stages

**Ejemplo: ModelSelectionStageAdapter**
```python
class ModelSelectionStageAdapter(PipelineStage):
    def get_stage_type(self) -> Stages:
        return Stages.MODEL_SELECTION
    
    def get_validator(self) -> Optional[StageValidator]:
        return ModelSelectionValidator()  # Define precondiciones
    
    def execute(self, context: RunContext) -> None:
        # Llama a la stage existente
        from app.core.stages.model_selection.model_selection_stage import ModelSelectionStage
        ModelSelectionStage(context=context).run()
```

---

### 5. StageExecution (Registro de Ejecución)
**Ubicación:** `app/core/orchestrator/stage_execution.py`

**Responsabilidades:**
- Rastrear toda la información de una ejecución
- Registrar timing, estado, errores, reintentos

**Campos:**
```python
@dataclass
class StageExecution:
    stage: Stages                        # Qué stage
    status: ExecutionStatus              # Estado (PENDING, RUNNING, SUCCESS, FAILED, SKIPPED)
    start_time: Optional[datetime]       # Cuándo empezó
    end_time: Optional[datetime]         # Cuándo terminó
    error: Optional[Exception]           # Qué error (si aplica)
    duration_seconds: float              # Cuánto tardó
    retry_count: int                     # Cuántos reintentos
    skip_reason: Optional[str]           # Por qué se saltó (si aplica)
```

---

### 6. ExecutionStatus (Estados de Ejecución)
**Ubicación:** `app/core/orchestrator/execution_status.py`

**Estados posibles:**
- **PENDING:** Stage está en cola, no ha empezado
- **RUNNING:** Stage está ejecutándose en este momento
- **SUCCESS:** Stage completó exitosamente
- **FAILED:** Stage falló con un error
- **SKIPPED:** Stage fue omitida porque sus precondiciones no se cumplieron

---

## Flujo de Ejecución

### Secuencia de Ejecución

```
1. main.py
   └─ Orchestrator(context).run()
      
      PARA CADA STAGE EN EL PIPELINE:
      
      ├─ 1. OBTENER INFORMACIÓN
      │  ├─ stage_type = stage.get_stage_type()
      │  ├─ validator = stage.get_validator()
      │  └─ Crear: execution = StageExecution(stage=stage_type)
      │
      ├─ 2. VALIDAR PRECONDICIONES
      │  ├─ Si validator existe:
      │  │  ├─ is_valid, msg = validator.validate(context)
      │  │  └─ Si NO es válido:
      │  │     ├─ execution.status = SKIPPED
      │  │     ├─ execution.skip_reason = msg
      │  │     └─ Continuar con siguiente stage
      │  │
      │  └─ Si validator es None o precondiciones OK:
      │     └─ Continuar
      │
      ├─ 3. EJECUTAR CON REINTENTOS
      │  ├─ Para intento = 1 hasta max_retries:
      │  │  ├─ execution.status = RUNNING
      │  │  ├─ stage.execute(context)  ← Código actual de la stage
      │  │  ├─ Si SUCCESS:
      │  │  │  ├─ execution.status = SUCCESS
      │  │  │  └─ Retornar ejecución
      │  │  │
      │  │  └─ Si ERROR:
      │  │     ├─ Si es último intento:
      │  │     │  ├─ execution.status = FAILED
      │  │     │  ├─ execution.error = exception
      │  │     │  └─ Retornar ejecución
      │  │     │
      │  │     └─ Si no es último intento:
      │  │        └─ Continuar al siguiente intento
      │
      ├─ 4. EJECUTAR EVALUACIÓN (si SUCCESS)
      │  └─ EvaluationStage(stage, context).run()
      │
      ├─ 5. RASTREAR EJECUCIÓN
      │  └─ self.executions.append(execution)
      │
      └─ SIGUIENTE STAGE

2. Mostrar Resumen
   ├─ Logs con estado de cada stage
   ├─ Contar: exitosas, fallidas, omitidas
   └─ Generar reporte JSON
```

---

## Diagrama de Conexiones

### Cómo se Conectan los Componentes

```
┌──────────────────────────────────────────────────────────────────┐
│                      main.py                                     │
│  orchestrator = Orchestrator(context, max_retries=2)            │
│  summary = orchestrator.run()                                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Orchestrator.run()                             │
│  _build_pipeline() → [Adapter1, Adapter2, Adapter3]             │
└──────────────────────┬───────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    ┌────────┐   ┌────────┐   ┌────────┐
    │Adapter1│   │Adapter2│   │Adapter3│
    └────┬───┘   └────┬───┘   └────┬───┘
         │            │            │
         │            │            │
    PARA CADA ADAPTER:
    
    ┌────────────────────────────────────────────────┐
    │  validator = adapter.get_validator()           │
    │                                                │
    │  Opciones:                                     │
    │  ├─ None (primera stage, sin precondiciones)  │
    │  └─ ValidatorInstance (requiere precondiciones)
    │                                                │
    │  Si validator ≠ None:                         │
    │    ├─ is_valid, msg = validator.validate()   │
    │    └─ Si NO válido → SKIPPED                  │
    │                                                │
    │  Si válido o sin precondiciones:              │
    │    ├─ adapter.execute(context)                │
    │    │  └─ Llama: OriginalStage(context).run()  │
    │    │                                           │
    │    └─ EvaluationStage.run() [si SUCCESS]      │
    └────────────────────────────────────────────────┘
```

### Flujo de Datos

```
RunContext (entrada)
    │
    ├─ config: Configuración del proyecto
    ├─ stage_results: Dict[Stages, StageResult]
    │   └─ Se actualiza después de cada stage
    │
    ▼ (Pasa a través de todas las stages)
    
RunContext (salida, actualizado)
    └─ stage_results está lleno con resultados de cada stage
```

---

## Guía de Mantenimiento

### Estructura de Carpetas

```
app/core/orchestrator/
├── __init__.py                  # Exporta API pública
├── orchestrator.py              # ⭐ Orquestador Principal
├── pipeline_stage.py            # Interfaz base de stages
├── stage_validator.py           # Interfaz base de validadores
├── stage_execution.py           # Modelo de datos de ejecución
├── execution_status.py          # Enum de estados
├── stages_adapters.py           # Adaptadores de stages existentes
├── validators.py                # Implementaciones de validadores
└── __pycache__/                 # Cachés compilados (ignorar)
```

### Flujo de Control de Calidad

Antes de hacer cambios:

1. **Entender el componente:**
   - Leer la docstring de la clase
   - Identificar sus responsabilidades
   - Ver cómo se conecta con otros componentes

2. **Hacer el cambio:**
   - Modificar solo lo necesario
   - Mantener la interfaz igual (no cambiar firmas de métodos)
   - Actualizar docstrings si cambia comportamiento

3. **Verificar:**
   - El código compila sin errores de sintaxis
   - Los imports funcionan
   - Las stages se ejecutan en orden
   - Se registran logs apropiadamente

---

## Cómo Agregar una Nueva Stage

### Paso 1: Crear la Clase de Stage Original

La stage debe heredar de tu clase base actual y implementar `run()`:

```python
# app/core/stages/new_feature/new_feature_stage.py

from app.core.context.run_context import RunContext

class NewFeatureStage:
    def __init__(self, context: RunContext):
        self.context = context
    
    def run(self):
        # Lógica aquí
        logger.info("Executing new feature stage...")
```

### Paso 2: Crear el Validador (si tiene precondiciones)

```python
# En app/core/orchestrator/validators.py

class NewFeatureValidator(StageValidator):
    def validate(self, context: RunContext) -> Tuple[bool, Optional[str]]:
        # Verificar precondiciones
        if Stages.REQUIRED_STAGE not in context.stage_results:
            return False, "REQUIRED_STAGE no completado"
        
        return True, None
```

### Paso 3: Crear el Adaptador

```python
# En app/core/orchestrator/stages_adapters.py

class NewFeatureStageAdapter(PipelineStage):
    def __init__(self, context: RunContext):
        self.context = context
    
    def get_stage_type(self) -> Stages:
        return Stages.NEW_FEATURE  # Debe estar definido en Stages enum
    
    def get_validator(self) -> Optional[StageValidator]:
        return NewFeatureValidator()  # O None si no tiene precondiciones
    
    def execute(self, context: RunContext) -> None:
        from app.core.stages.new_feature.new_feature_stage import NewFeatureStage
        
        logger.debug("Executing new feature stage...")
        NewFeatureStage(context=context).run()
```

### Paso 4: Registrar el Nuevo Enum

```python
# app/core/context/stages.py

class Stages(Enum):
    DATA_HANDLER = "data_handler"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    NEW_FEATURE = "new_feature"        # ← Nueva stage
    FINE_TUNING = "fine_tuning"
```

### Paso 5: Agregar al Pipeline

```python
# app/core/orchestrator/orchestrator.py

def _build_pipeline(self) -> List[PipelineStage]:
    return [
        DataInspectionStageAdapter(self.context),
        FeatureSelectionStageAdapter(self.context),
        ModelSelectionStageAdapter(self.context),
        NewFeatureStageAdapter(self.context),      # ← Nueva línea
        FineTuningStageAdapter(self.context),
    ]
```

---

## Cómo Modificar Componentes Existentes

### Caso 1: Cambiar la Lógica de una Stage Existente

✓ **Lo que debes hacer:**
- Modificar directamente el archivo de la stage original
- Ejemplo: `app/core/stages/feature_selection/feature_selection_stage.py`
- El adaptador automáticamente usará la nueva versión

✗ **Lo que NO debes hacer:**
- Modificar el adaptador
- Cambiar el validador (a menos que cambien las precondiciones)

### Caso 2: Cambiar las Precondiciones de una Stage

✓ **Pasos:**
1. Modificar el validador correspondiente en `validators.py`
2. Documentar el cambio (qué precondiciones cambiaron)
3. Probar que la validación funciona correctamente

Ejemplo:
```python
class ModelSelectionValidator(StageValidator):
    def validate(self, context: RunContext) -> Tuple[bool, Optional[str]]:
        # ANTES: Solo verificaba FEATURE_SELECTION
        # AHORA: También verifica que haya suficientes features seleccionadas
        
        if Stages.FEATURE_SELECTION not in context.stage_results:
            return False, "FEATURE_SELECTION no completado"
        
        # Nueva precondición
        if not context.stage_results[Stages.FEATURE_SELECTION].metadata.get('selector'):
            return False, "FEATURE_SELECTION no seleccionó features"
        
        return True, None
```

### Caso 3: Cambiar el Orden de Ejecución

✓ **Lo que debes hacer:**
- Modificar `_build_pipeline()` en `orchestrator.py`
- Reordenar los adaptadores en el orden deseado

Ejemplo:
```python
def _build_pipeline(self) -> List[PipelineStage]:
    return [
        DataInspectionStageAdapter(self.context),
        ModelSelectionStageAdapter(self.context),    # Movido
        FeatureSelectionStageAdapter(self.context),  # Movido
        FineTuningStageAdapter(self.context),
    ]
```

⚠️ **Importante:** Los validadores deben reflejar el nuevo orden de dependencias.

### Caso 4: Agregar Reintentos Diferenciados

✓ **Modificar en `orchestrator.py`:**

```python
# En lugar de usar el mismo max_retries para todas:

def _execute_with_retry(self, stage: PipelineStage, execution: StageExecution) -> StageExecution:
    stage_type = stage.get_stage_type()
    
    # Reintentos por tipo de stage
    retries_for_stage = {
        Stages.DATA_HANDLER: 3,      # Más reintentos para lectura de datos
        Stages.MODEL_SELECTION: 2,
        Stages.FEATURE_SELECTION: 1,
    }
    
    max_retries = retries_for_stage.get(stage_type, self.max_retries)
    
    # ... resto del código
```

---

## Resolución de Problemas

### Problema: "Stage fue omitida (SKIPPED)"

**Causas posibles:**
1. Stage anterior no completó correctamente
2. Stage anterior no produjo resultados esperados
3. Validador rechazó las precondiciones

**Solución:**
1. Ver logs para encontrar el mensaje de precondición fallida
2. Verificar que la stage anterior (dependencia) completó exitosamente
3. Revisar el validador para entender qué espera

Ejemplo de log:
```
[feature_selection] Precondition failed: DATA_HANDLER not completed
⊘ FEATURE_SELECTION | Skipped | Reason: DATA_HANDLER not completed
```

---

### Problema: Stage Falla Repetidamente

**Causas posibles:**
1. Error de lógica en la stage
2. Datos inválidos del contexto
3. Recurso externo no disponible (temporalmente)

**Solución:**
1. Ver el error específico en los logs
2. Aumentar `max_retries` si es error temporal
3. Revisar la stage original para corregir la lógica

```python
# Aumentar reintentos (en main.py)
orchestrator = Orchestrator(context, max_retries=5)  # De 2 a 5
```

---

### Problema: No Hay Logs de una Stage

**Causas posibles:**
1. Stage nunca fue ejecutada (omitida o error anterior)
2. Logger no está configurado correctamente
3. Adaptador no está en el pipeline

**Solución:**
1. Verificar que la stage está en `_build_pipeline()`
2. Verificar que no fue omitida por precondiciones
3. Revisar que el logger está importado correctamente

---

### Problema: Reporte de Ejecución Vacío

**Causas posibles:**
1. `run()` no fue llamado
2. No hay stages en el pipeline
3. Exception no capturada

**Solución:**
```python
# Verificar que se llama a run()
summary = orchestrator.run()

# Ver el reporte
print(orchestrator.get_execution_report_json())

# O ver ejecuciones directamente
for exec in orchestrator.executions:
    print(f"{exec.stage.value}: {exec.status.value}")
```

---

## Checklist de Cambios Seguros

Cuando hagas cambios, verifica:

- [ ] El código compila sin errores de sintaxis
- [ ] Los imports son correctos
- [ ] Las firmas de métodos no cambiaron (si es interfaz)
- [ ] La documentación está actualizada
- [ ] Los logs son informativos
- [ ] Las precondiciones son claras y correctas
- [ ] El pipeline ejecuta en orden correcto
- [ ] Se registran todas las ejecuciones
- [ ] El reporte JSON es generado correctamente
- [ ] Probaste con datos de prueba

---

## Resumen Visual

```
COMPONENTES Y SUS ROLES:

ExecutionStatus     → Define estados posibles (enum)
    ↑
StageExecution      → Registra información de ejecución (dataclass)
    ↑
StageValidator      → Valida precondiciones (interfaz)
    ↑
PipelineStage       → Define contrato de stage (interfaz)
    ↑
StagesAdapters      → Adaptan stages existentes (concretas)
    ↑
Orchestrator        → Coordina todo el proceso (director)
    ↑
main.py             → Punto de entrada
```

---

**Documento generado:** 2026-03-13
**Versión:** 1.0
**Estado:** Documentación Interna

