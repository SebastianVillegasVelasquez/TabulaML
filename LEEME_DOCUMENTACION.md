# Documentación del Sistema - Índice

## 📚 Documentación Disponible

Este directorio contiene documentación interna del proyecto TabulaML enfocada en el sistema de orquestación del pipeline.

### 📄 Documentos Principales

#### 1. **DOCUMENTACION_ORCHESTRATOR.md** ⭐
**Lectura recomendada:** PRIMERO

Documentación completa y detallada del sistema de orquestación.

**Contenido:**
- Visión general del sistema
- Arquitectura de capas
- Descripción de cada componente
- Flujo de ejecución paso a paso
- Diagramas de conexión
- Guía de mantenimiento
- Cómo agregar nuevas stages
- Cómo modificar componentes existentes
- Resolución de problemas

**Cuándo leer:** Cuando necesites entender cómo funciona todo el sistema, cómo está organizado, y cómo mantenerlo.

---

#### 2. **GUIA_RAPIDA_ORCHESTRATOR.md** ⚡
**Lectura recomendada:** SEGUNDO (como referencia rápida)

Guía de referencia rápida para desarrolladores.

**Contenido:**
- Estructura rápida del código
- Cambios comunes (cómo hacer cosas típicas)
- Resolución rápida de problemas
- Estados de ejecución
- Propiedades disponibles
- Ejemplos de uso
- Checklist de seguridad

**Cuándo usar:** Cuando necesitas hacer un cambio específico y quieres recordar exactamente qué archivos tocar y cómo hacerlo.

---

#### 3. **PATRONES_MANTENIMIENTO_ORCHESTRATOR.md** 🔧
**Lectura recomendada:** TERCERO (como referencia para cambios avanzados)

Patrones comunes para mantenimiento y extensión del sistema.

**Contenido:**
- Patrón: Agregar logging detallado
- Patrón: Métricas personalizadas
- Patrón: Reintentos específicos por stage
- Patrón: Validaciones complejas
- Patrón: Comportamiento post-ejecución
- Patrón: Testing de stages
- Patrón: Eventos/Callbacks

**Cuándo usar:** Cuando necesitas hacer un cambio más complejo o quieres ver ejemplos de cómo se extiende el sistema.

---

## 🎯 Flujo de Lectura Recomendado

### Para Nuevo Desarrollador
1. Leer: **DOCUMENTACION_ORCHESTRATOR.md** - Sección "Visión General" y "Arquitectura"
2. Leer: **DOCUMENTACION_ORCHESTRATOR.md** - Sección "Componentes Principales"
3. Leer: **GUIA_RAPIDA_ORCHESTRATOR.md** - Secciones "Estructura Rápida" y "Cambios Comunes"
4. Explorar: Código en `app/core/orchestrator/`

### Para Hacer un Cambio Específico
1. Consultar: **GUIA_RAPIDA_ORCHESTRATOR.md** - Sección "Cambios Comunes"
2. Si necesitas más detalle: **DOCUMENTACION_ORCHESTRATOR.md** - Sección relevante
3. Si es cambio avanzado: **PATRONES_MANTENIMIENTO_ORCHESTRATOR.md**

### Para Resolver un Problema
1. Buscar en: **GUIA_RAPIDA_ORCHESTRATOR.md** - Sección "Resolución Rápida de Problemas"
2. Si no encuentra: **DOCUMENTACION_ORCHESTRATOR.md** - Sección "Resolución de Problemas"
3. Ver logs y contexto específico del proyecto

---

## 🏗️ Estructura del Código

```
app/core/orchestrator/
├── __init__.py                  # Exporta API pública
├── orchestrator.py              # ⭐ Orquestador Principal
├── pipeline_stage.py            # Interfaz base de stages
├── stage_validator.py           # Interfaz base de validadores
├── stage_execution.py           # Modelo de datos de ejecución
├── execution_status.py          # Enum de estados
├── stages_adapters.py           # Adaptadores de stages existentes
└── validators.py                # Implementaciones de validadores
```

---

## 🔍 Resumen de Componentes

| Componente | Propósito | Cambio Frecuencia |
|-----------|----------|-------------------|
| **Orchestrator** | Coordina ejecución | Rara |
| **PipelineStage** | Interfaz de stages | Nunca (es interfaz) |
| **StageValidator** | Interfaz de validadores | Nunca (es interfaz) |
| **StagesAdapters** | Adaptan stages existentes | Rara (solo agregar nuevas) |
| **Validators** | Implementan validaciones | Frecuente (cambiar lógica) |
| **StageExecution** | Registra métricas | Rara (extender campos) |
| **ExecutionStatus** | Define estados | Nunca (enum fijo) |

---

## 📝 Cómo Usar Esta Documentación

### Caso 1: Entender Cómo Funciona el Sistema
→ Leer: `DOCUMENTACION_ORCHESTRATOR.md` completo

### Caso 2: Cambiar Lógica de una Stage Existente
→ Consultar: `GUIA_RAPIDA_ORCHESTRATOR.md` → "Cambiar Lógica de una Stage"
→ Luego: Editar directamente el archivo de la stage

### Caso 3: Agregar una Nueva Stage
→ Consultar: `DOCUMENTACION_ORCHESTRATOR.md` → "Cómo Agregar una Nueva Stage"
→ O usar: `GUIA_RAPIDA_ORCHESTRATOR.md` → "Agregar Nueva Stage"

### Caso 4: Cambiar Precondiciones
→ Consultar: `GUIA_RAPIDA_ORCHESTRATOR.md` → "Cambiar Precondiciones"
→ Luego: Editar archivo `validators.py`

### Caso 5: Implementar Funcionalidad Avanzada
→ Consultar: `PATRONES_MANTENIMIENTO_ORCHESTRATOR.md`
→ Encontrar patrón similar al que necesitas
→ Adaptar según contexto del proyecto

---

## 🚨 Importante

⚠️ **Estos documentos NO se suben al repositorio** (están en `.gitignore`)

Motivos:
- Son documentación interna de desarrollo
- Pueden cambiar frecuentemente
- No son parte del código publicable
- Sirven solo como referencia local

✅ **Sí se incluye:** Documentación en código (docstrings, comentarios)
✅ **Sí se incluye:** README.md general del proyecto
✅ **Sí se incluye:** Documentación de API (si existe)

---

## 📌 Checklist: Antes de Hacer Cambios

Use estos documentos para verificar que está haciendo el cambio correctamente:

- [ ] Consulté el documento relevante
- [ ] Identifiqué exactamente qué archivo debo cambiar
- [ ] Entiendo las implicaciones del cambio
- [ ] Revisé ejemplos similares en el código
- [ ] Probé localmente antes de hacer commit
- [ ] La documentación está actualizada (si aplica)
- [ ] Los tests pasan (si existen)

---

## 🤝 Contribución

Si necesitas:

1. **Aclarar conceptos:** Edita `DOCUMENTACION_ORCHESTRATOR.md`
2. **Agregar patrón común:** Edita `PATRONES_MANTENIMIENTO_ORCHESTRATOR.md`
3. **Mejorar guía rápida:** Edita `GUIA_RAPIDA_ORCHESTRATOR.md`

Mantén los documentos:
- ✓ Actualizados con cambios del código
- ✓ Claros y concisos
- ✓ Con ejemplos reales
- ✓ Organizados por temas

---

**Índice de Documentación**
**Versión:** 1.0
**Última actualización:** 2026-03-13
**Estado:** Interna - No publicada

