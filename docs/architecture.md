# 🏗️ Arquitectura del Proyecto

Este documento explica las decisiones de diseño y la estructura del servicio ML.

## 🎯 Decisiones de Diseño

### ¿Por qué un servicio independiente?

He decidido separar el ML en un servicio independiente por varias razones:

1. **Separación de responsabilidades**
   - El equipo de desarrollo web no necesita entender ML
   - Yo puedo iterar en modelos sin afectar la app web
   - Código más limpio y mantenible

2. **Deploy independiente**
   - Puedo actualizar modelos sin redesplegar toda la app
   - Escalado independiente si hay mucha carga de predicciones
   - Diferentes stacks tecnológicos (ML vs Web)

3. **Colaboración del equipo**
   - Repos separados = menos conflictos en git
   - Cada uno trabaja en su área sin interferir
   - Integración clara vía API REST

### ¿Por qué stateless?

Un servicio **stateless** significa que no tiene estado/memoria entre requests:
- ❌ NO tiene base de datos
- ❌ NO guarda información de usuarios
- ❌ NO mantiene sesiones
- ✅ Solo recibe datos, procesa y responde

**Ventajas:**
- Más fácil de escalar (puedes levantar múltiples instancias)
- Sin problemas de consistencia de datos
- Más simple de mantener
- Deploy más rápido

