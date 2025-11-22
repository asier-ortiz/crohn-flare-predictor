# Mockups HTML/CSS - Crohn Tracker

Mockups estáticos de alta fidelidad para la aplicación web de seguimiento de Enfermedad Inflamatoria Intestinal (EII).

## 📁 Archivos Incluidos

- **landing.html** - Página principal pública (marketing)
- **login.html** - Login y registro de usuarios
- **dashboard.html** - Panel principal con predicciones e insights
- **diario.html** - Formulario de registro diario de síntomas
- **reportes.html** - Historial y gráficas de evolución
- **cuenta.html** - Gestión de perfil y configuración

## 🚀 Cómo Usar

### Opción 1: Abrir directamente en el navegador

```bash
# Desde la terminal
open docs/mockups/landing.html

# O simplemente hacer doble click en cualquier archivo .html
```

Los mockups se abren directamente en tu navegador sin necesidad de servidor.

### Opción 2: Tomar capturas de pantalla

```bash
# Mac: Cmd + Shift + 4
# Windows: Win + Shift + S
# Linux: PrtScn o Shift + PrtScn
```

### Opción 3: Servidor local (opcional)

Si quieres un servidor HTTP local:

```bash
cd docs/mockups
python3 -m http.server 8080
# Abre http://localhost:8080 en tu navegador
```

## 📊 Estructura de Navegación

```
Landing Page (landing.html)
    ↓
Login/Registro (login.html)
    ↓
Dashboard (dashboard.html) ← Página principal tras login
    ├─→ Diario (diario.html)
    ├─→ Reportes (reportes.html)
    └─→ Cuenta (cuenta.html)
```

## 🎨 Tecnologías Utilizadas

- **HTML5** - Estructura semántica
- **Tailwind CSS** (vía CDN) - Estilos y diseño responsive
- **Chart.js** (vía CDN) - Gráficas interactivas
- **Font Awesome** (vía CDN) - Iconos

**Nota**: Todos los recursos se cargan desde CDN, por lo que necesitas conexión a internet para ver los estilos correctamente.

## 🎯 Características

### Landing Page
- Hero section con CTA
- Sección "Cómo Funciona" (3 pasos)
- Características destacadas
- Estadísticas del modelo ML
- Footer completo

### Login/Registro
- Tabs para alternar entre login y registro
- Integración con Google/GitHub (UI solamente)
- Validación de fortaleza de contraseña (visual)
- Diseño centrado y responsive

### Dashboard
- **6 zonas principales**:
  1. Estado Actual (círculo de riesgo)
  2. Alertas y Recomendaciones
  3. Gráfica de Evolución (30 días)
  4. Insights de Lifestyle (3 cards)
  5. Resumen Mensual (4 métricas)
  6. Acciones Rápidas
- Gráfica interactiva con Chart.js
- Datos de ejemplo realistas

### Diario
- Sliders interactivos para síntomas (0-10)
- Colores dinámicos según valor
- Gestión de alimentos (añadir/eliminar)
- Selector de ejercicio (3 opciones)
- Botones "Guardar" y "Guardar y Predecir"

### Reportes
- Filtros (período, vista, métrica)
- Gráfica de evolución dual (riesgo + síntomas)
- Tabla paginada con datos históricos
- 3 tarjetas de insights del período
- Botón "Exportar a PDF"

### Cuenta
- Formulario de perfil personal
- Datos médicos completos (según API)
- Gestión de medicamentos
- Configuración de notificaciones
- Zona de peligro (eliminar cuenta)

## 📱 Responsive Design

Todos los mockups son **mobile-first** y se adaptan a:

- **Desktop** (>1024px) - Layout completo con sidebar/grid
- **Tablet** (768px-1024px) - Layout adaptado, algunas columnas se apilan
- **Mobile** (<768px) - Stack vertical, hamburger menu (solo UI)

Para ver en móvil, abre en navegador y redimensiona la ventana o usa DevTools (F12 → Toggle Device Toolbar).

## 🎨 Paleta de Colores

```css
/* Primario */
--blue-500: #3B82F6;    /* Botones, enlaces */
--blue-600: #2563EB;    /* Hover */

/* Estados */
--green-500: #10B981;   /* Riesgo bajo, éxito */
--yellow-500: #F59E0B;  /* Riesgo moderado, warning */
--red-500: #EF4444;     /* Riesgo alto, error */

/* Fondo */
--gray-50: #F9FAFB;     /* Fondo general */
--gray-100: #F3F4F6;    /* Cards, inputs */
```

## 🔧 Interactividad

Los mockups incluyen **JavaScript básico** para:

- Alternar entre tabs (Login/Registro)
- Actualizar valores de sliders en tiempo real
- Renderizar gráficas con Chart.js
- Cambiar colores según valores de síntomas

**Nota**: No incluyen funcionalidad real de backend. Son solo mockups visuales.

## 📦 Para Presentar

Si necesitas presentar los mockups:

### Opción A: Screenshots
1. Abre cada HTML en pantalla completa
2. Toma captura (Cmd+Shift+4 en Mac)
3. Organiza en un PDF o presentación

### Opción B: PDF desde navegador
1. Abre cualquier HTML
2. Cmd+P (Imprimir)
3. "Guardar como PDF"
4. Ajusta márgenes y orientación

### Opción C: Entregar HTMLs directamente
Simplemente comparte la carpeta `docs/mockups/` completa. Cualquiera puede abrirlos sin instalación.

## 📋 Checklist de Funcionalidades Visualizadas

- [x] Sistema de predicción de riesgo (bajo/moderado/alto)
- [x] Gráficas de evolución temporal
- [x] Registro diario de síntomas (7 campos)
- [x] Seguimiento de alimentos (texto libre)
- [x] Niveles de ejercicio (3 opciones)
- [x] Insights de lifestyle (triggers, beneficiosos, ejercicio)
- [x] Alertas y recomendaciones médicas
- [x] Historial con tabla paginada
- [x] Gestión de perfil médico completo
- [x] Gestión de medicamentos
- [x] Configuración de notificaciones
- [x] Exportar a PDF (botón)
- [x] Responsive design (mobile/tablet/desktop)

## 💡 Notas para Desarrollo

Estos mockups sirven como **especificación visual** para el desarrollo en Vue.js:

1. **Estructura HTML** → Componentes Vue
2. **Clases Tailwind** → Mantener igual en Vue + Tailwind
3. **Chart.js** → Usar vue-chartjs o similar
4. **Interacciones** → Implementar en Vue con v-model, @click, etc.
5. **Datos de ejemplo** → Reemplazar con llamadas a la API `/predict?format=simple`

## 🔗 Integración con la API

Los mockups están diseñados para consumir la API del proyecto:

```javascript
// Ejemplo de integración
const response = await fetch('/predict?format=simple', {
  method: 'POST',
  body: JSON.stringify(dailyRecords)
});

const data = await response.json();

// Mapear respuesta a componentes del Dashboard
dashboard.risk = data.risk;
dashboard.alerts = data.alerts;
dashboard.lifestyle_tips = data.lifestyle_tips;
```

Ver `docs/DATABASE_SCHEMA.md` y `docs/WEB_INTERFACE.md` para más detalles de integración.

## ✅ Validación

Los mockups cumplen con:

- ✅ Especificación de `docs/WEB_INTERFACE.md`
- ✅ Paleta de colores definida
- ✅ 6 zonas del dashboard implementadas
- ✅ Wireframes ASCII convertidos a HTML real
- ✅ Datos de ejemplo basados en la API real
- ✅ Responsive y mobile-friendly
- ✅ Accesibilidad básica (contraste, tamaños)

## 📞 Soporte

Para modificar los mockups:

1. Abre el archivo HTML en un editor de texto
2. Busca la sección que quieres cambiar
3. Modifica las clases de Tailwind o el contenido
4. Guarda y recarga en el navegador

**Tailwind CSS Docs**: https://tailwindcss.com/docs

---

Creado para el proyecto **Crohn Flare Predictor** - Mockups de alta fidelidad sin backend
