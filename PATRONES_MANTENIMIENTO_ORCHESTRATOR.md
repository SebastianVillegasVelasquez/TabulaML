# Patrones Comunes de Mantenimiento del Orchestrator

## 📌 Tabla de Contenidos
1. [Patrón: Agregar Logging Detallado](#patrón-agregar-logging-detallado)
2. [Patrón: Metricas Personalizadas](#patrón-métricas-personalizadas)
3. [Patrón: Reintentos Específicos por Stage](#patrón-reintentos-específicos-por-stage)
4. [Patrón: Validaciones Complejas](#patrón-validaciones-complejas)
5. [Patrón: Modificar Comportamiento Post-Ejecución](#patrón-modificar-comportamiento-post-ejecución)
6. [Patrón: Testing de Stages](#patrón-testing-de-stages)

---

## Patrón: Agregar Logging Detallado

### Problema
Se necesita agregar logs más detallados para debugging.

### Solución

**En el adaptador:**
```python
class ModelSelectionStageAdapter(PipelineStage):
    def execute(self, context: RunContext) -> None:
        logger.debug(f"ModelSelection: Using {len(context.config.scoring)} metrics")
        
        from app.core.stages.model_selection.model_selection_stage import ModelSelectionStage
        ModelSelectionStage(context=context).run()
        
        # Log del resultado
        result = context.stage_results[self.get_stage_type()]
        if result.best_experiment:
            logger.debug(
                f"ModelSelection best: {result.best_experiment.name} "
                f"with score {result.best_experiment.metrics}"
            )
```

---

## Patrón: Métricas Personalizadas

### Problema
Se necesita capturar métricas adicionales de cada stage (ej: memoria usada, datos procesados).

### Solución

**Extender StageExecution:**
```python
# En stage_execution.py
@dataclass
class StageExecution:
    # ... campos existentes ...
    
    # Agregar campos personalizados
    memory_used_mb: Optional[float] = None
    records_processed: Optional[int] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
```

**En el orchestrator:**
```python
# En orchestrator.py, en _execute_with_retry
import psutil

execution.start_time = datetime.now()
process = psutil.Process()
mem_before = process.memory_info().rss / 1024 / 1024

stage.execute(self.context)

mem_after = process.memory_info().rss / 1024 / 1024
execution.memory_used_mb = mem_after - mem_before
```

---

## Patrón: Reintentos Específicos por Stage

### Problema
Algunas stages necesitan más reintentos que otras (datos pueden ser temporalmente inaccesibles, etc.).

### Solución

**Crear configuración de reintentos:**
```python
# En orchestrator.py, al principio de la clase

RETRY_CONFIG = {
    Stages.DATA_HANDLER: 3,        # Más reintentos para I/O
    Stages.FEATURE_SELECTION: 2,
    Stages.MODEL_SELECTION: 1,     # Menos para modelo
}

def __init__(self, context: RunContext, max_retries: int = 2):
    # ...
    self.max_retries = max_retries
```

**Usar en _execute_with_retry:**
```python
def _execute_with_retry(self, stage: PipelineStage, execution: StageExecution):
    stage_type = stage.get_stage_type()
    max_retries = self.RETRY_CONFIG.get(stage_type, self.max_retries)
    
    for attempt in range(1, max_retries + 1):
        # ... resto del código
```

---

## Patrón: Validaciones Complejas

### Problema
Una stage necesita validaciones más complejas que solo "¿existe la anterior?".

### Solución

**Crear validador específico:**
```python
# En validators.py

class AdvancedFeatureSelectionValidator(StageValidator):
    def validate(self, context: RunContext) -> Tuple[bool, Optional[str]]:
        # Validar existencia
        if Stages.DATA_HANDLER not in context.stage_results:
            return False, "DATA_HANDLER no completado"
        
        # Validar cantidad de features
        data_result = context.stage_results[Stages.DATA_HANDLER]
        if not data_result.results:
            return False, "DATA_HANDLER sin resultados"
        
        # Validar config
        if not context.config.scoring:
            return False, "No hay métricas configuradas"
        
        # Validar memoria disponible (ejemplo)
        import psutil
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        if available_memory < 1000:  # Menos de 1GB
            return False, "Memoria insuficiente"
        
        return True, None
```

**Usar en adaptador:**
```python
class FeatureSelectionStageAdapter(PipelineStage):
    def get_validator(self) -> Optional[StageValidator]:
        return AdvancedFeatureSelectionValidator()
```

---

## Patrón: Modificar Comportamiento Post-Ejecución

### Problema
Después de ejecutar una stage, se necesita hacer algo especial (limpiar archivos temporales, actualizar BD, etc.).

### Solución

**Usar un hook post-ejecución:**
```python
# En orchestrator.py

def _post_stage_execution(self, stage: PipelineStage, execution: StageExecution) -> None:
    """Ejecutar acciones después de que una stage completa."""
    
    if execution.status == ExecutionStatus.SUCCESS:
        stage_type = execution.stage
        
        # Ejemplo: Limpiar archivos temporales
        if stage_type == Stages.MODEL_SELECTION:
            logger.info("Limpiando archivos temporales de model selection...")
            # limpiar archivos
        
        # Ejemplo: Guardar snapshot del contexto
        logger.info(f"Guardando snapshot después de {stage_type.value}")
        # guardar snapshot
```

**Llamar en run():**
```python
def run(self) -> Dict[str, Any]:
    # ...
    for stage in self._pipeline:
        # ... validar y ejecutar ...
        
        if execution.status == ExecutionStatus.SUCCESS:
            self._post_stage_execution(stage, execution)
            self._run_evaluation(stage_type)
```

---

## Patrón: Testing de Stages

### Problema
Se necesita probar stages individualmente sin ejecutar todo el pipeline.

### Solución

**Crear tests unitarios:**
```python
# tests/unit/test_model_selection_stage.py

import pytest
from unittest.mock import Mock, patch
from app.core.orchestrator.stages_adapters import ModelSelectionStageAdapter
from app.core.orchestrator.validators import ModelSelectionValidator
from app.core.context.stages import Stages

class TestModelSelectionStageAdapter:
    
    def test_get_stage_type(self):
        context = Mock()
        adapter = ModelSelectionStageAdapter(context)
        assert adapter.get_stage_type() == Stages.MODEL_SELECTION
    
    def test_get_validator(self):
        context = Mock()
        adapter = ModelSelectionStageAdapter(context)
        validator = adapter.get_validator()
        assert isinstance(validator, ModelSelectionValidator)
    
    def test_validator_requires_feature_selection(self):
        context = Mock()
        context.stage_results = {}
        
        validator = ModelSelectionValidator()
        is_valid, msg = validator.validate(context)
        
        assert not is_valid
        assert "FEATURE_SELECTION" in msg

    def test_validator_passes_with_feature_selection(self):
        context = Mock()
        context.stage_results = {
            Stages.FEATURE_SELECTION: Mock()
        }
        
        validator = ModelSelectionValidator()
        is_valid, msg = validator.validate(context)
        
        assert is_valid
        assert msg is None
```

**Test de integración:**
```python
# tests/integration/test_orchestrator_flow.py

def test_orchestrator_executes_pipeline_in_order(mock_context):
    from app.core.orchestrator import Orchestrator
    
    orchestrator = Orchestrator(mock_context, max_retries=1)
    summary = orchestrator.run()
    
    # Verificar que se ejecutaron en orden
    stage_names = [exec.stage.value for exec in orchestrator.executions]
    assert stage_names == [
        "data_handler",
        "feature_selection",
        "model_selection"
    ]

def test_orchestrator_skips_stage_without_preconditions(mock_context):
    # Mock: DATA_HANDLER no completó
    mock_context.stage_results = {}
    
    from app.core.orchestrator import Orchestrator
    
    orchestrator = Orchestrator(mock_context, max_retries=1)
    summary = orchestrator.run()
    
    # FEATURE_SELECTION debe estar en skipped
    assert "feature_selection" in summary["skipped"]
```

---

## Patrón: Agregar Eventos/Callbacks

### Problema
Se necesita ejecutar código cuando ocurren ciertos eventos (antes de stage, después de validación, etc.).

### Solución

**Crear sistema de eventos:**
```python
# En orchestrator.py

from typing import Callable

class Orchestrator:
    def __init__(self, context: RunContext, max_retries: int = 2):
        # ...
        self.callbacks = {
            'before_stage': [],
            'after_validation': [],
            'after_execution': [],
            'on_error': []
        }
    
    def on(self, event: str, callback: Callable) -> None:
        """Registrar callback para evento."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_event(self, event: str, **kwargs) -> None:
        """Ejecutar callbacks de evento."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                logger.error(f"Error en callback {event}: {e}")

    def run(self):
        # ... en el loop principal ...
        
        for stage in self._pipeline:
            self._trigger_event('before_stage', stage=stage)
            
            # ... validación ...
            self._trigger_event('after_validation', stage=stage, valid=is_valid)
            
            # ... ejecución ...
            self._trigger_event('after_execution', stage=stage, execution=execution)
```

**Usar:**
```python
# En main.py

def log_stage_start(stage):
    print(f"🚀 Iniciando: {stage.get_stage_type().value}")

def log_stage_complete(stage, execution):
    print(f"✓ Completó: {stage.get_stage_type().value}")

orchestrator = Orchestrator(context, max_retries=2)
orchestrator.on('before_stage', log_stage_start)
orchestrator.on('after_execution', log_stage_complete)
summary = orchestrator.run()
```

---

## 🎯 Checklist: Antes de Hacer Cambios

- [ ] Entiendo qué componente afecta el cambio
- [ ] He identificado dónde hacer el cambio
- [ ] He revisado ejemplos existentes en el código
- [ ] He actualizado la documentación
- [ ] He probado localmente
- [ ] Los tests pasan
- [ ] He verificado que no rompe cambios anteriores

---

**Documento:** Patrones de Mantenimiento
**Versión:** 1.0
**Última actualización:** 2026-03-13

