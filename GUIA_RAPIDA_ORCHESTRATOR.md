# Guía Rápida del Orchestrator - Referencia Rápida

## 🚀 Para Empezar

### Ejecutar el Pipeline
```python
from app.core.orchestrator import Orchestrator

orchestrator = Orchestrator(context, max_retries=2)
summary = orchestrator.run()

# Ver resultado
print(orchestrator.get_execution_report_json())
```

---

## 📊 Estructura Rápida

### Archivos Principales
| Archivo | Propósito |
|---------|-----------|
| `orchestrator.py` | Orquestador principal |
| `pipeline_stage.py` | Interfaz base de stages |
| `stage_validator.py` | Interfaz base de validadores |
| `stages_adapters.py` | Adaptadores de stages existentes |
| `validators.py` | Implementaciones de validadores |
| `stage_execution.py` | Registro de ejecución |
| `execution_status.py` | Estados posibles |

---

## 🔧 Cambios Comunes

### ✏️ Cambiar Lógica de una Stage
1. Editar: `app/core/stages/[stage_name]/[stage_name].py`
2. No modificar el adaptador
3. Probar: `python app/main.py`

### ➕ Agregar Nueva Stage
1. Crear: `app/core/stages/new_stage/new_stage.py`
2. Crear Enum: Agregar a `app/core/context/stages.py`
3. Crear Validador: Agregar a `app/core/orchestrator/validators.py`
4. Crear Adaptador: Agregar a `app/core/orchestrator/stages_adapters.py`
5. Registrar: Agregar a `_build_pipeline()` en `orchestrator.py`

### 🔄 Cambiar Precondiciones
1. Editar validador en: `app/core/orchestrator/validators.py`
2. Actualizar lógica de validación
3. Probar: `python app/main.py`

### 🎯 Cambiar Orden de Ejecución
1. Editar: `orchestrator.py` → `_build_pipeline()`
2. Reordenar adaptadores en lista
3. Verificar que validadores reflejen nuevas dependencias

---

## 🆘 Resolución Rápida de Problemas

### Stage Omitida (SKIPPED)
→ Ver logs para mensaje de precondición fallida
→ Verificar que stage anterior completó

### Stage Falla
→ Aumentar `max_retries`: `Orchestrator(context, max_retries=5)`
→ Ver error en logs

### Sin Logs
→ Verificar: ¿Stage en pipeline? ¿No omitida?
→ Check logger está importado

### Reporte Vacío
→ Verificar: `orchestrator.run()` fue llamado?
→ `orchestrator.executions` tiene elementos?

---

## 📝 Estados de Ejecución

```
PENDING  → En cola
RUNNING  → En ejecución
SUCCESS  → Completó bien
FAILED   → Falló con error
SKIPPED  → Omitida (precondiciones no cumplidas)
```

---

## 📊 Propiedades de StageExecution

```python
execution.stage              # Qué stage
execution.status             # Estado actual
execution.start_time         # Cuándo empezó
execution.end_time           # Cuándo terminó
execution.duration_seconds   # Cuánto tardó
execution.error              # Excepción (si falló)
execution.retry_count        # Reintentos
execution.skip_reason        # Por qué se saltó

# Propiedades convenientes
execution.is_complete        # ¿Terminó? (SUCCESS, FAILED o SKIPPED)
execution.was_successful     # ¿Exitosa?
execution.was_skipped        # ¿Fue omitida?
```

---

## 💡 Ejemplos de Uso

### Acceder a Resultados
```python
summary = orchestrator.run()

# Por tipo de estado
print(f"Exitosas: {summary['success']}")
print(f"Fallidas: {summary['failed']}")
print(f"Omitidas: {summary['skipped']}")
```

### Verificar una Stage Específica
```python
for execution in orchestrator.executions:
    if execution.stage == Stages.FEATURE_SELECTION:
        print(f"Status: {execution.status.value}")
        print(f"Duration: {execution.duration_seconds:.2f}s")
        if execution.error:
            print(f"Error: {execution.error}")
```

### Exportar Reporte
```python
# Como JSON
report_json = orchestrator.get_execution_report_json()
with open('execution_report.json', 'w') as f:
    f.write(report_json)

# Como diccionarios Python
report_dicts = orchestrator.get_execution_report()
for report in report_dicts:
    print(report)
```

---

## ⚠️ Recuerda

- ✓ Los adaptadores NO se modifican para cambiar lógica
- ✓ Las precondiciones van en `validators.py`
- ✓ Las nuevas stages se agregan a `_build_pipeline()`
- ✓ El pipeline se ejecuta en orden definido
- ✓ Si una stage falla, las siguientes pueden ser omitidas
- ✓ Siempre revisar los logs para debugging

---

**Última actualización:** 2026-03-13

