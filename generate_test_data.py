import random
from pathlib import Path
from pump_problem import generate_random_instance

# --- Configuración de rutas ---
BASE_DATA_DIR = Path("data")
TEST_DIR = BASE_DATA_DIR / "test"

# --- Configuración de la generación ---
NUM_PUMPS = 3
HORIZON = 8
NUM_INSTANCES = 10


def main():
    # 1. Crear el directorio de destino si no existe
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Generando instancias en: {TEST_DIR.resolve()}\n")

    # Regla operativa: Dejar al menos 1 bomba de reserva (backup).
    # Calculamos esto ANTES para pasarlo al generador.
    max_active = max(1, NUM_PUMPS - 1)

    failed_count = 0

    # 2. Generar los problemas
    for i in range(1, NUM_INSTANCES + 1):
        # Patrón de nombre: problem_{bombas}_{horizonte}_{indice}
        p_id = f"problem_{NUM_PUMPS}_{HORIZON}_{i:03d}"

        # Semilla única por instancia para reproducibilidad
        rng = random.Random(i)

        try:
            # Pasamos la restricción explícitamente AL generador.
            # Esto asegura que la demanda generada sea satisfacible con 'max_active' bombas.
            problem = generate_random_instance(
                problem_id=p_id,
                num_pumps=NUM_PUMPS,
                horizon=HORIZON,
                max_active_pumps=max_active,  # <--- CLAVE: Restricción aplicada durante la creación
                rng=rng
            )

            # Guardar a disco
            file_path = TEST_DIR / f"{p_id}.json"
            problem.save(file_path)

            print(f"✅ [{p_id}] Guardado -> {file_path.name}")

        except RuntimeError as e:
            print(f"❌ [{p_id}] Falló la generación: {e}")
            failed_count += 1

    # 3. Resumen final
    success_count = NUM_INSTANCES - failed_count
    print(f"\n🎉 Generación completada.")
    print(f"   Exitosas: {success_count}/{NUM_INSTANCES}")
    print(f"   Fallidas: {failed_count}/{NUM_INSTANCES}")


if __name__ == "__main__":
    main()