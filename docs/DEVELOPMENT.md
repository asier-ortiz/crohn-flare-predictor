# 💻 Guía de Desarrollo

Instrucciones para trabajar en el código ML de este proyecto.

## 🚀 Setup Inicial

### 1. Clonar el repositorio

```bash
git clone <repo-url>
cd crohn-flare-predictor
```

### 2. Instalar uv (si no lo tienes)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# O con pip
pip install uv
```

### 3. Instalar dependencias

```bash
# Para desarrollo completo (API + notebooks + ML)
uv sync --group dev --group notebooks

# Solo para correr la API
uv sync
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env con tus valores
```

### 5. Iniciar el servicio

```bash
# Opción 1: Con make
make serve

# Opción 2: Directo
uv run uvicorn api.app:app --reload --port 8001

# Opción 3: Con Docker
docker-compose up
```

## 📁 Flujo de Trabajo

### Desarrollo de Modelos ML

1. **Análisis exploratorio** en notebooks

```bash
# Iniciar Jupyter
make notebook

# O directo
uv run jupyter notebook
```

2. **Entrenamiento** en `notebooks/03_model_training.ipynb`

3. **Guardar modelo** en `models/crohn_predictor.pkl`

4. **Actualizar código** en `src/model.py`

5. **Probar** la API con el nuevo modelo

### Desarrollo de API

1. **Modificar** `api/app.py` o `api/schemas.py`

2. **Probar** cambios:
```bash
# La API se recarga automáticamente con --reload
# Prueba en http://localhost:8001/docs
```

3. **Tests**:
```bash
make test
# O
uv run pytest
```

4. **Formatear código**:
```bash
make format
# O
uv run black api/ src/
```

## 🧪 Testing

### Tests unitarios

```bash
# Todos los tests
uv run pytest

# Con cobertura
uv run pytest --cov=src --cov=api

# Un archivo específico
uv run pytest tests/test_model.py
```

### Probar la API

```bash
# Asegúrate de que el servidor esté corriendo
make serve

# En otra terminal
make test-api

# O manualmente
curl http://localhost:8001/health
```

## 📊 Trabajar con Notebooks

### Instalar kernel de Jupyter

```bash
uv run python -m ipykernel install --user --name=crohn-ml
```

### Notebooks disponibles

1. `01_exploratory_analysis.ipynb` - Análisis de datos
2. `02_feature_engineering.ipynb` - Creación de features
3. `03_model_training.ipynb` - Entrenamiento de modelos

### Mejores prácticas

- ✅ Ejecuta celdas en orden
- ✅ Limpia outputs antes de commit
- ✅ Documenta decisiones importantes
- ✅ Guarda visualizaciones importantes

## 🔧 Comandos Útiles

```bash
make help          # Ver todos los comandos
make dev           # Setup completo de desarrollo
make serve         # Iniciar API
make test          # Ejecutar tests
make format        # Formatear código
make lint          # Verificar código
make clean         # Limpiar archivos temporales
make test-api      # Probar endpoints
```

## 🐛 Debugging

### Logs

```bash
# La API muestra logs en consola
# Nivel de log configurable en .env (LOG_LEVEL=DEBUG)
```

### Python debugger

```python
# En cualquier parte del código
import pdb; pdb.set_trace()
```

### VS Code

```json
// .vscode/launch.json
{
    "configurations": [
        {
            "name": "FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "api.app:app",
                "--reload",
                "--port",
                "8001"
            ]
        }
    ]
}
```

## 📦 Gestión de Dependencias

### Añadir dependencia

```bash
# Producción
uv add pandas

# Desarrollo
uv add --group dev pytest

# Notebooks
uv add --group notebooks matplotlib
```

### Actualizar dependencias

```bash
uv lock --upgrade
uv sync
```

## 🐳 Docker

### Build

```bash
docker build -t crohn-ml-api .
```

### Run

```bash
docker run -p 8001:8001 crohn-ml-api
```

### Con docker-compose

```bash
docker-compose up --build
```

## 🔄 Git Workflow

```bash
# 1. Crear rama para feature
git checkout -b feature/mejora-modelo

# 2. Hacer cambios y commits
git add .
git commit -m "Mejora precisión del modelo"

# 3. Push
git push origin feature/mejora-modelo

# 4. Crear PR en GitHub
```

## 📝 Convenciones de Código

### Python

- Usar **Black** para formato
- Líneas máximo 100 caracteres
- Type hints siempre que sea posible
- Docstrings para funciones públicas

### Commits

```
tipo(scope): descripción corta

Descripción más larga si es necesario

Ejemplos:
- feat(model): añadir modelo XGBoost
- fix(api): corregir validación de síntomas
- docs(readme): actualizar instrucciones
- refactor(preprocessing): simplificar pipeline
```

## 🚨 Troubleshooting

### Error: "Module not found"

```bash
# Reinstalar dependencias
uv sync --reinstall
```

### Error: "Port already in use"

```bash
# Matar proceso en puerto 8001
lsof -ti:8001 | xargs kill -9
```

### Error: "ML model not loaded"

```bash
# Verifica que existe models/crohn_predictor.pkl
ls -la models/

# O usa el modelo de reglas (por defecto)
```

## 💡 Tips

1. **Hot reload**: Usa `--reload` para que la API se recargue con cambios
2. **IPython**: Para probar código rápido: `uv run ipython`
3. **Logs**: Aumenta verbosity con `LOG_LEVEL=DEBUG`
4. **Cache**: Limpia con `make clean` si hay problemas raros

## 📞 Ayuda

- Issues del proyecto en GitHub
- Documentación de FastAPI: https://fastapi.tiangolo.com
- Documentación de uv: https://docs.astral.sh/uv
