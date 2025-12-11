import random
from pathlib import Path
from pump_problem import generate_random_instance, WaterPumpProblem

# --- Configuración de rutas ---
BASE_DATA_DIR = Path("data")
TEST_DIR = BASE_DATA_DIR / "test"

# --- Configuración de la generación ---
NUM_PUMPS = 3
HORIZON = 10
NUM_INSTANCES = 10


def main():
    # 1. Crear el directorio de destino si no existe
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Generando instancias en: {TEST_DIR.resolve()}\n")

    # 2. Generar los problemas
    for i in range(1, NUM_INSTANCES + 1):

        # --- CAMBIO AQUÍ: Nuevo formato de nombre ---
        # Patrón: problem_{bombas}_{horizonte}_{indice}
        p_id = f"problem_{NUM_PUMPS}_{HORIZON}_{i:03d}"

        # Generamos una semilla única basada en el índice para reproducibilidad
        rng = random.Random(i)

        try:
            problem = generate_random_instance(
                problem_id=p_id,
                num_pumps=NUM_PUMPS,
                horizon=HORIZON,
                rng=rng
            )

            # --- Ajuste Lógico para pocas bombas ---
            # Con solo 3 bombas, reducimos el máximo activo a 2
            # para mantener la restricción de "1 de reserva".
            problem.max_active_pumps = max(1, NUM_PUMPS - 1)

            # Guardar a disco
            file_path = TEST_DIR / f"{p_id}.json"
            problem.save(file_path)

            # Feedback por consola
            print(f"✅ [{p_id}] Guardado -> {file_path.name}")

        except RuntimeError as e:
            print(f"❌ [{p_id}] Falló la generación: {e}")

    print("\n🎉 Generación completada.")


if __name__ == "__main__":
    main()