# Referencia Rápida - Nuevo Sistema de Evaluación

## 📋 Estructura

```
app/core/stages/evaluation/
├── base_evaluator.py                    ← Clase base (no editar)
├── feature_selection_evaluator.py       ← Lógica Feature Selection
├── model_selection_evaluator.py         ← Lógica Model Selection
├── evaluator_factory.py                 ← Factory (crea evaluadores)
├── evaluation_stage.py                  ← Orquestador (simplificado)
├── evaluator.py                         ← Selector mejor experimento
└── model_registry.py                    ← Persistencia modelos
```

---

## 🎯 Flujo (En 3 Pasos)

```
1. Stage.run()
   └─ Ejecuta experimentos → Guarda resultados RAW

2. EvaluationStage.run()
   ├─ Obtiene experimentos
   ├─ Selecciona mejor
   └─ Crea evaluador vía Factory
        └─ evaluator.evaluate(experiments)

3. Evaluador específico
   ├─ _extract_stage_specific_data()
   └─ _update_context()
```

---

## 🚀 Agregar Nueva Stage

### 1. Crear Evaluador

```python
# app/core/stages/evaluation/new_stage_evaluator.py

from app.core.stages.evaluation.base_evaluator import BaseEvaluator

class NewStageEvaluator(BaseEvaluator):
    def _extract_stage_specific_data(self, sorted_results, best):
        # Tu lógica aquí
        return {
            'key1': value1,
            'key2': value2,
        }
    
    def _update_context(self, sorted_results, best, data):
        # Actualizar contexto aquí
        pass
```

### 2. Registrar

```python
# En evaluator_factory.py o main.py

from app.core.stages.evaluation.new_stage_evaluator import NewStageEvaluator

EvaluatorFactory.register(Stages.NEW_STAGE, NewStageEvaluator)
```

### 3. ¡Listo!
EvaluationStage funciona automáticamente.

---

## 📚 Métodos Disponibles en BaseEvaluator

Usa estos métodos en tus evaluadores:

```python
# Ordenar por métrica principal
sorted_results = self._sort_results(results)

# Extraer top-k por familia de modelo
top_models = self._extract_top_k_by_family(sorted_results, k=3)

# Obtener familia de un modelo
family = self._get_model_family(experiment)

# Obtener familia de un selector
selector_name = experiment.config.get('selector')

# Log del mejor experimento
self._log_best_experiment(best_experiment)
```

---

## 🎯 Model Selection: Top-3 de Diferentes Familias

```python
# Automático con:
top_models = self._extract_top_k_by_family(sorted_results, k=3)

# Resultado:
{
    'RandomForest': best_rf,    # Mejor de su familia
    'XGBoost': best_xgb,        # Mejor de su familia
    'LogisticRegression': best_lr  # Mejor de su familia
}
```

---

## ✅ Checklist: Antes de Usar

- [ ] BaseEvaluator no editar
- [ ] Crear evaluador heredando BaseEvaluator
- [ ] Implementar `_extract_stage_specific_data()`
- [ ] Implementar `_update_context()`
- [ ] Registrar con `EvaluatorFactory.register()`
- [ ] Probar con: `python app/main.py`

---

## 📊 Comparativa de Responsabilidades

| Componente | Responsabilidad |
|-----------|-----------------|
| **Stage** | Ejecutar experimentos |
| **EvaluationStage** | Orquestar (delegar) |
| **BaseEvaluator** | Lógica común + template |
| **[Stage]Evaluator** | Lógica específica |
| **Factory** | Crear evaluador correcto |

---

## 🔍 Debugging

### Ver qué evaluador se crea

```python
# En EvaluationStage.run()
evaluator = EvaluatorFactory.create(self.stage, self.context)
print(f"Evaluator: {evaluator.__class__.__name__}")  # FeatureSelectionEvaluator
```

### Ver datos extraídos

```python
# En tu evaluador
data = self._extract_stage_specific_data(sorted_results, best)
print(f"Data: {data}")  # Ver todos los datos
```

---

## 🎓 Ejemplos Reales

### Feature Selection
```python
class FeatureSelectionEvaluator(BaseEvaluator):
    def _extract_stage_specific_data(self, sorted_results, best):
        top_k = self._extract_top_k_selectors(sorted_results, k=3)
        features = self._extract_feature_data(best)
        return {'top_k_selectors': top_k, **features}
```

### Model Selection
```python
class ModelSelectionEvaluator(BaseEvaluator):
    def _extract_stage_specific_data(self, sorted_results, best):
        top_models = self._extract_top_k_by_family(sorted_results, k=3)
        return {'top_k_models_by_family': top_models}
```

---

## ❓ Preguntas Frecuentes

**P: ¿Debo editar BaseEvaluator?**
R: No. Solo crea subclases que hereden de él.

**P: ¿Cómo agrego nueva stage?**
R: 1 clase + 1 línea registro. Ver "Agregar Nueva Stage" arriba.

**P: ¿Dónde va la lógica específica?**
R: En `_extract_stage_specific_data()` de tu evaluador.

**P: ¿Cómo actualizo contexto?**
R: En `_update_context()` de tu evaluador.

**P: ¿Qué si necesito data común?**
R: Usa métodos de BaseEvaluator o crea método helper.

---

**Última actualización:** 2026-03-13
**Status:** Listo para usar ✅

