# Guía de Interfaz Web - Crohn Flare Predictor

Documento conceptual para el diseño de la aplicación web de seguimiento de EII (Enfermedad Inflamatoria Intestinal).

---

## Estructura General de la Aplicación

La aplicación web consta de **6 páginas principales**:

1. **Landing Page** - Página pública informativa
2. **Login / Registro** - Autenticación de usuarios
3. **Dashboard** - Panel principal con insights y predicción
4. **Diario** - Registro diario de síntomas, alimentos y ejercicio
5. **Reportes** - Historial de predicciones y evolución temporal
6. **Cuenta** - Gestión de perfil y medicamentos

---

## Páginas Principales

### 1. Landing Page (Pública)

**Objetivo**: Informar sobre la aplicación y captar nuevos usuarios.

**Contenido**:
- **Hero Section**: Título llamativo + descripción breve + CTA "Comenzar gratis"
  - Ejemplo: "Predice brotes de Crohn con inteligencia artificial"
- **Cómo Funciona** (3 pasos):
  1. Registra tus síntomas diarios
  2. La IA analiza patrones y tendencias
  3. Recibe predicciones y recomendaciones personalizadas
- **Características Destacadas**:
  - Predicción basada en ML (92.6% precisión)
  - Seguimiento de alimentos y ejercicio
  - Insights personalizados sobre triggers
  - Exporta reportes para tu médico
- **Testimonios** (opcional)
- **Footer**: Enlaces legales, contacto, redes sociales

**Diseño**: Limpio, profesional, colores suaves (azul/verde). Evitar elementos que asocien con "enfermedad".

---

### 2. Login / Registro

**Objetivo**: Autenticación segura y onboarding simple.

#### Login
- Email
- Contraseña
- "Recordarme"
- "¿Olvidaste tu contraseña?"
- Botón "Iniciar Sesión"

#### Registro
- Email
- Contraseña (con indicador de fortaleza)
- Nombre completo
- Aceptar términos y condiciones
- Botón "Crear Cuenta"

**Onboarding después del registro**:

Wizard de 3 pasos para completar el perfil médico:

**Paso 1: Datos Demográficos**
- Edad
- Género (M/F/Otro)
- Peso y altura (para calcular BMI)

**Paso 2: Información Médica**
- Tipo de EII: Crohn o Colitis Ulcerosa
- Años desde diagnóstico
- Clasificación de Montreal (dropdown con explicaciones)
- ¿Has tenido cirugías? (Sí/No)
- Estado de fumador (Nunca/Ex-fumador/Fumador)

**Paso 3: Historial de Brotes y Medicamentos**
- Número de brotes previos
- ¿Hace cuántos días fue tu último brote? (aproximado)
- Días acumulados en brote (aproximado)
- Medicamentos actuales (campo de texto libre + autocompletado)

**Importante**: Este wizard solo se muestra una vez. Después, los datos se pueden editar en "Cuenta".

---

### 3. Dashboard (Página Principal)

**Objetivo**: Mostrar de un vistazo el estado actual del paciente, predicción de riesgo, tendencias y recomendaciones.

El dashboard está dividido en **6 zonas principales**:

---

#### Zona 1: Estado Actual (Hero del Dashboard)

**Posición**: Arriba a la izquierda (zona más visible)

**Contenido**:
- **Nivel de Riesgo Actual**:
  - Círculo grande con color (verde/amarillo/rojo)
  - Texto: "Riesgo Bajo" / "Riesgo Moderado" / "Riesgo Alto"
  - Puntuación numérica: "3.5/10"
- **Mensaje Principal**:
  - "Riesgo bajo de brote en los próximos 7 días"
- **Fecha de la predicción**:
  - "Última actualización: 22 Nov 2025"
- **Botón CTA**:
  - "Actualizar Predicción" (lleva a Diario si faltan datos del día)

**Diseño Visual**:
- Fondo con gradiente suave según riesgo
- Verde (#10B981) para bajo
- Amarillo (#F59E0B) para moderado
- Rojo (#EF4444) para alto

---

#### Zona 2: Alertas y Recomendaciones

**Posición**: Debajo del Estado Actual o a la derecha

**Contenido**:
- **Recomendación Principal** (desde la API):
  - Ícono de doctor/médico
  - Texto personalizado: "Monitoree sus síntomas de cerca. Considere contactar a su médico si empeoran."
- **Alertas** (si existen, desde `alerts` de la API):
  - Lista de alertas con íconos de advertencia
  - Ejemplo: "⚠️ Sangre en heces reportada en la última semana"
  - Ejemplo: "⚠️ Escalada rápida de síntomas detectada"

**Diseño Visual**:
- Tarjeta con borde suave
- Alertas con fondo amarillo claro si son warnings
- Fondo rojo claro si son críticas

---

#### Zona 3: Gráfica de Evolución

**Posición**: Centro del dashboard (ocupa buen espacio horizontal)

**Contenido**:
- **Gráfica de línea** que muestra:
  - Eje X: Últimos 30 días
  - Eje Y: Score de riesgo (0-10)
  - Línea de tendencia suave
  - Puntos clickeables (tooltip con detalles del día)
- **Indicador de tendencia**:
  - Flecha hacia arriba (rojo): "Tus síntomas están empeorando"
  - Flecha hacia abajo (verde): "Tus síntomas están mejorando"
  - Flecha horizontal (gris): "Tus síntomas están estables"

**Interactividad**:
- Al hacer hover sobre un punto, mostrar tooltip:
  - Fecha
  - Score de riesgo
  - Síntomas principales del día
- Botón "Ver Historial Completo" (lleva a Reportes)

**Diseño Visual**:
- Librería de gráficas: Chart.js o Recharts (Vue)
- Colores consistentes con la paleta de riesgo

---

#### Zona 4: Insights Personalizados (Lifestyle)

**Posición**: Debajo de la gráfica o a la derecha

**Contenido**:
- **Tarjetas de Insights de Alimentos y Ejercicio** (desde `lifestyle_tips` de la API)

**Estructura de cada tarjeta**:

**Tarjeta 1: Alimentos a Reducir**
- Ícono: ⚠️
- Título: "Alimentos que pueden estar afectándote"
- Lista de triggers:
  - "⚠️ Lácteos: correlación 0.51 con síntomas"
  - "⚠️ Café: correlación 0.48 con síntomas"
- Consejo: "Considera reducir el consumo de estos alimentos"

**Tarjeta 2: Alimentos Beneficiosos**
- Ícono: ✅
- Título: "Alimentos que te ayudan"
- Lista de beneficios:
  - "✅ Verduras: correlación -0.64 (reducen síntomas)"
  - "✅ Proteínas: correlación -0.52 (reducen síntomas)"
- Consejo: "Intenta incluir más de estos alimentos"

**Tarjeta 3: Ejercicio**
- Ícono: 🏃
- Título: "Impacto del ejercicio"
- Dato principal: "El ejercicio se asocia con reducción del 56% en severidad de síntomas"
- Consejo: "✅ Mantén el ejercicio: se asocia con menos síntomas"

**Diseño Visual**:
- Tarjetas pequeñas (cards) con bordes suaves
- Iconos grandes y visibles
- Fondo verde claro para beneficios
- Fondo amarillo claro para warnings

---

#### Zona 5: Resumen Mensual (Stats)

**Posición**: Barra inferior o sidebar

**Contenido**:
- **4 métricas clave en formato de mini-cards**:

  1. **Días Analizados**: "7 días"
  2. **Días con Riesgo Alto**: "2 días este mes"
  3. **Adherencia al Registro**: "85%" (días registrados / días totales)
  4. **Promedio de Severidad**: "4.2/10"

**Diseño Visual**:
- 4 tarjetas pequeñas en fila horizontal (desktop) o 2x2 (mobile)
- Cada tarjeta con ícono, número grande y descripción
- Colores neutros (gris/azul claro)

---

#### Zona 6: Acciones Rápidas

**Posición**: Barra lateral derecha o botones flotantes

**Contenido**:
- **Botón Principal**: "Registrar Día de Hoy" (va a Diario)
- **Botón Secundario**: "Exportar Reporte PDF" (descarga PDF con gráfica + datos)
- **Botón Terciario**: "Ver Historial Completo" (va a Reportes)

**Diseño Visual**:
- Botones con íconos claros
- Primario: Color azul destacado
- Secundarios: Colores neutros

---

### Layout del Dashboard (Desktop)

```
+-------------------------------------------------------------+
|  NAVBAR: Logo | Dashboard | Diario | Reportes | Cuenta      |
+-------------------------------------------------------------+
|                                                             |
|  +-----------------------+  +-----------------------------+ |
|  |   ZONA 1:             |  |   ZONA 2:                   | |
|  |   Estado Actual       |  |   Alertas y                 | |
|  |   (Riesgo Bajo)       |  |   Recomendaciones           | |
|  |   3.5/10              |  |                             | |
|  +-----------------------+  +-----------------------------+ |
|                                                             |
|  +-------------------------------------------------------+  |
|  |   ZONA 3: Gráfica de Evolución (30 días)             |  |
|  |                                                       |  |
|  |   [Gráfica de línea con tendencia]                   |  |
|  +-------------------------------------------------------+  |
|                                                             |
|  +-------------------+  +-------------------+  +---------+  |
|  | ZONA 4:           |  | ZONA 4:           |  | ZONA 6: |  |
|  | Insights          |  | Insights          |  | Acciones|  |
|  | Alimentos Trigger |  | Ejercicio         |  | Rápidas |  |
|  +-------------------+  +-------------------+  +---------+  |
|                                                             |
|  +-------------------------------------------------------+  |
|  |   ZONA 5: Resumen Mensual (4 métricas)               |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

---

### Layout del Dashboard (Mobile)

En móvil, las zonas se apilan verticalmente:

1. Estado Actual (full width)
2. Alertas y Recomendaciones (full width)
3. Gráfica de Evolución (full width, scrollable horizontalmente)
4. Insights (cards apiladas verticalmente)
5. Resumen Mensual (2x2 grid)
6. Acciones Rápidas (botones flotantes en la parte inferior)

---

## 4. Página de Diario

**Objetivo**: Permitir al usuario registrar síntomas, alimentos y ejercicio diarios de forma rápida y sencilla.

### Estructura

**Selector de Fecha**
- Por defecto: Hoy
- Permite seleccionar fecha pasada (últimos 30 días)

**Formulario dividido en 3 secciones**:

---

#### Sección 1: Síntomas

7 campos con **sliders visuales** (0-10):

1. **Dolor Abdominal**: Slider de 0 (sin dolor) a 10 (máximo dolor)
2. **Diarrea**: Slider de 0 (normal) a 10 (muy frecuente)
3. **Fatiga**: Slider de 0 (sin cansancio) a 10 (muy cansado)
4. **Náuseas**: Slider de 0 (sin náuseas) a 10 (muy nauseabundo)

3 campos de tipo **checkbox/toggle**:

5. **Fiebre**: Sí/No
6. **Sangre en Heces**: Sí/No
7. **Cambio de Peso**: Campo numérico (kg ganados/perdidos)

**Diseño Visual**:
- Sliders con colores: verde (0-3), amarillo (4-6), rojo (7-10)
- Checkboxes grandes y fáciles de tocar (mobile-friendly)

---

#### Sección 2: Alimentos

**Campo de entrada de alimentos**:
- Input de texto libre con **autocompletado** (usa diccionario de alimentos comunes)
- Botón "+ Añadir Alimento"
- Lista de alimentos añadidos (con botón "X" para eliminar)

**Ejemplo**:
```
┌─────────────────────────────────────┐
│  ¿Qué comiste hoy?                  │
│  ┌───────────────────────────────┐  │
│  │ café con leche            [+] │  │
│  └───────────────────────────────┘  │
│                                     │
│  Alimentos añadidos:                │
│  • café con leche          [X]      │
│  • tostadas                [X]      │
│  • ensalada cesar          [X]      │
└─────────────────────────────────────┘
```

**Nota para implementación**:
- El backend categoriza automáticamente (lácteos, gluten, etc.)
- No es necesario que el frontend haga categorización
- Solo envía el texto tal cual a la API

---

#### Sección 3: Ejercicio

**Selector simple de 3 opciones** (botones grandes):

```
┌───────────────────────────────────────────────────┐
│  ¿Hiciste ejercicio hoy?                          │
│                                                   │
│  [  Sin ejercicio  ]  [  Moderado  ]  [  Intenso  ] │
│         🛋️                🚶               🏃         │
└───────────────────────────────────────────────────┘
```

- **Sin ejercicio**: Botón por defecto
- **Moderado**: Caminata, yoga, ejercicio suave
- **Intenso**: Running, gimnasio, ejercicio intenso

---

#### Botones de Acción

- **Guardar**: Guarda el registro en la BD
- **Guardar y Predecir**: Guarda el registro y redirige al Dashboard con nueva predicción

**Validación**:
- Al menos los síntomas numéricos deben estar completados
- Los alimentos y ejercicio son opcionales

---

## 5. Página de Reportes

**Objetivo**: Visualizar el historial completo de predicciones y tendencias a largo plazo.

### Contenido

#### Filtros
- Rango de fechas (últimos 7 días, 30 días, 3 meses, 6 meses, todo)
- Tipo de vista: Gráfica / Tabla

#### Vista de Gráfica
- **Gráfica de línea dual**:
  - Línea 1: Score de riesgo (0-10)
  - Línea 2: Severidad de síntomas (0-10)
- Permite comparar predicción vs síntomas reales

#### Vista de Tabla
- Tabla con columnas:
  - Fecha
  - Riesgo (badge con color)
  - Score
  - Tendencia
  - Alimentos Trigger
  - Acciones (Ver Detalles)

#### Exportar a PDF
- Botón "Exportar Reporte" genera PDF con:
  - Gráfica de evolución
  - Tabla de datos
  - Insights principales
  - Recomendaciones
  - Logo y fecha del reporte

**Uso**: El paciente puede llevar este PDF a su consulta médica.

---

## 6. Página de Cuenta

**Objetivo**: Gestionar perfil, medicamentos y configuración.

### Secciones

#### Mi Perfil
- Nombre completo
- Email
- Contraseña (cambiar)

#### Datos Médicos
- Formulario editable con los mismos campos del onboarding:
  - Edad, género, BMI
  - Tipo de EII, clasificación Montreal
  - Historial de brotes
  - Cirugías, estado de fumador

#### Mis Medicamentos
- Lista de medicamentos actuales (con botón "Eliminar")
- Botón "+ Añadir Medicamento"
- Input de texto libre con autocompletado

#### Configuración
- Idioma (Español/English)
- Notificaciones por email (Sí/No)
- Recordatorios diarios (Sí/No, hora)

#### Eliminar Cuenta
- Botón rojo "Eliminar mi cuenta"
- Modal de confirmación con advertencia

---

## Paleta de Colores Recomendada

### Colores Principales
- **Primario (Azul)**: #3B82F6 - Botones, enlaces, navbar
- **Secundario (Verde)**: #10B981 - Éxito, riesgo bajo
- **Advertencia (Amarillo)**: #F59E0B - Warnings, riesgo moderado
- **Error (Rojo)**: #EF4444 - Alertas críticas, riesgo alto

### Colores de Fondo
- **Fondo principal**: #F9FAFB (gris muy claro)
- **Tarjetas**: #FFFFFF (blanco)
- **Hover**: #F3F4F6 (gris claro)

### Texto
- **Primario**: #111827 (casi negro)
- **Secundario**: #6B7280 (gris medio)
- **Terciario**: #9CA3AF (gris claro)

---

## Interacción con la API

### Endpoints Utilizados

#### 1. Registro de Síntomas (Diario)
- **Acción**: Usuario guarda registro del día
- **Flujo**:
  1. Frontend guarda datos en la BD (tabla `daily_records` y `daily_foods`)
  2. Si el usuario hace click en "Guardar y Predecir":
     - Frontend hace POST a `/predict?format=simple`
     - Envía últimos 7-14 días de datos
     - Recibe respuesta simplificada
     - Guarda predicción en tabla `predictions`
     - Redirige a Dashboard con datos actualizados

#### 2. Dashboard
- **Acción**: Usuario entra al dashboard
- **Flujo**:
  1. Frontend obtiene última predicción de la BD (tabla `predictions`)
  2. Si la última predicción tiene más de 24 horas:
     - Muestra opción "Actualizar predicción"
     - Al hacer click, hace POST a `/predict?format=simple`
  3. Muestra datos de la respuesta simplificada:
     - `risk.level` y `risk.score` → Zona 1
     - `recommendation` → Zona 2
     - `alerts` → Zona 2
     - `trend.description` → Zona 3
     - `lifestyle_tips` → Zona 4
     - `summary.days_analyzed` → Zona 5

#### 3. Reportes
- **Acción**: Usuario ve historial
- **Flujo**:
  1. Frontend obtiene últimas 30-90 predicciones de la BD
  2. Dibuja gráfica con esos datos
  3. Permite exportar a PDF (generación en frontend con jsPDF o similar)

### Respuesta de la API (formato simple)

```json
{
  "risk": {
    "level": "medium",
    "level_es": "moderado",
    "score": 5.5,
    "message": "Riesgo moderado de brote en los próximos 7 días"
  },
  "recommendation": "Monitoree sus síntomas de cerca. Considere contactar a su médico si empeoran.",
  "trend": {
    "direction": "worsening",
    "direction_es": "empeorando",
    "description": "Tus síntomas están empeorando"
  },
  "alerts": [
    "Severidad alta de síntomas en días recientes",
    "Sangre en heces reportada en la última semana"
  ],
  "lifestyle_tips": [
    "⚠️ Considera reducir lácteos: correlación 0.51 con síntomas",
    "✅ Aumenta consumo de verduras: correlación inversa -0.64",
    "✅ Mantén el ejercicio: se asocia con 56% menos síntomas"
  ],
  "summary": {
    "date": "2025-11-22",
    "days_analyzed": 7,
    "period": "15/11/2025 - 22/11/2025"
  }
}
```

---

## Consideraciones de UX

### Accesibilidad
- Contraste de colores WCAG AA compliant
- Tamaños de fuente mínimos (16px para texto)
- Botones grandes (mínimo 44x44px para mobile)
- Textos alternativos en todas las imágenes

### Mobile-First
- Diseñar primero para móvil, luego adaptar a desktop
- Sliders grandes y fáciles de arrastrar
- Botones flotantes para acciones principales

### Onboarding
- Tour guiado la primera vez que el usuario entra al Dashboard
- Tooltips explicativos en campos del formulario médico
- Mensajes de ayuda contextual

### Estados Vacíos
- **Dashboard sin datos**: "Registra tu primer día para obtener insights personalizados"
- **Reportes sin historial**: "Necesitas al menos 7 días de registros para ver tendencias"

### Loading States
- Skeleton screens mientras carga el dashboard
- Spinners en botones al guardar/predecir
- Progress bar al generar PDF

---

## Notas de Implementación para el Equipo

### Priorización
1. **Fase 1** (MVP): Login, Diario, Dashboard básico (sin lifestyle insights)
2. **Fase 2**: Lifestyle insights, gráfica de evolución
3. **Fase 3**: Reportes avanzados, exportar PDF
4. **Fase 4**: Notificaciones, recordatorios

### Librerías Recomendadas (Vue.js)
- **UI Framework**: Vuetify o PrimeVue (componentes pre-diseñados)
- **Gráficas**: Chart.js con vue-chartjs
- **Formularios**: VeeValidate (validación)
- **Fechas**: Day.js (más ligero que Moment.js)
- **HTTP**: Axios
- **Estado**: Pinia (Vuex está deprecated)
- **PDF**: jsPDF + html2canvas (para exportar)

### API Integration
- Usar `?format=simple` para todas las llamadas desde el frontend
- Cachear la última predicción en localStorage (evitar llamadas innecesarias)
- Mostrar datos cached mientras se actualiza en background

### Testing
- Probar con datos del endpoint `/predict` en Swagger
- Usar los ejemplos de test que están en `/tmp/test_*.json`

---

## Wireframes Conceptuales

### Dashboard (Desktop)
```
┌────────────────────────────────────────────────────────────┐
│  NAVBAR: 🏥 Crohn Tracker | Dashboard | Diario | Reportes  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────┐  ┌──────────────────────────────┐   │
│  │  RIESGO BAJO     │  │  RECOMENDACIÓN               │   │
│  │                  │  │  Continúe con el seguimiento │   │
│  │      3.5         │  │  regular...                  │   │
│  │     ──────       │  │                              │   │
│  │      10          │  │  ALERTAS:                    │   │
│  │                  │  │  ⚠️ Sangre en heces última   │   │
│  │  🟢 (círculo)    │  │     semana                   │   │
│  │                  │  │                              │   │
│  └──────────────────┘  └──────────────────────────────┘   │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  EVOLUCIÓN (30 DÍAS)                               │   │
│  │                                                    │   │
│  │     10 │                              •••         │   │
│  │      8 │                          ••••            │   │
│  │      6 │                     •••••                │   │
│  │      4 │              •••••••                     │   │
│  │      2 │         ••••••                           │   │
│  │      0 └──────────────────────────────────────    │   │
│  │         1   5   10  15  20  25  30 (días)        │   │
│  │                                                    │   │
│  │  ↗ Tus síntomas están empeorando                  │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────┐    │
│  │ ⚠️ TRIGGERS    │  │ ✅ BENEFICIOS  │  │ 🏃 EJERCI │    │
│  │                │  │                │  │          │    │
│  │ Lácteos (0.51) │  │ Verduras       │  │ Reducción│    │
│  │ Café (0.48)    │  │ (-0.64)        │  │ 56% en   │    │
│  │                │  │                │  │ síntomas │    │
│  │ Reduce consumo │  │ Aumenta consumo│  │          │    │
│  └────────────────┘  └────────────────┘  └──────────┘    │
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  7 DÍAS ANALIZADOS | 2 DÍAS ALTO | 85% ADHERENCIA │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  [🖊️ Registrar Día de Hoy]  [📄 Exportar PDF]            │
└────────────────────────────────────────────────────────────┘
```

### Diario (Mobile)
```
┌────────────────────┐
│  📅 Hoy            │
├────────────────────┤
│                    │
│  SÍNTOMAS          │
│                    │
│  Dolor Abdominal   │
│  ●───────○─────  5 │
│                    │
│  Diarrea           │
│  ●─────────○───  7 │
│                    │
│  Fatiga            │
│  ●────○────────  3 │
│                    │
│  Náuseas           │
│  ○─────────────  0 │
│                    │
│  ☐ Fiebre          │
│  ☑ Sangre en heces │
│                    │
│  Peso: -0.5 kg     │
│                    │
├────────────────────┤
│  ALIMENTOS         │
│                    │
│  [Buscar alimento] │
│                    │
│  • café con leche  │
│  • tostadas        │
│                    │
├────────────────────┤
│  EJERCICIO         │
│                    │
│  [Sin ejercicio]   │
│  [✓ Moderado]      │
│  [Intenso]         │
│                    │
├────────────────────┤
│  [Guardar]         │
│  [Guardar y        │
│   Predecir]        │
└────────────────────┘
```

---

## Flujo de Usuario Típico

1. **Usuario nuevo**:
   - Landing → Registro → Onboarding (3 pasos) → Dashboard vacío → "Registra tu primer día"

2. **Usuario recurrente (día 1-6)**:
   - Login → Dashboard (sin trends aún) → Diario → Registra síntomas

3. **Usuario con datos suficientes (día 7+)**:
   - Login → Dashboard con predicción completa → Ve insights → Ajusta dieta/ejercicio → Mejora síntomas

4. **Antes de consulta médica**:
   - Login → Reportes → Exportar PDF → Lleva a consulta

---

Esta guía proporciona una visión completa y conceptual de la interfaz web. El equipo de frontend puede usar este documento como referencia para implementar cada página y componente sin necesidad de especificaciones técnicas de código.
