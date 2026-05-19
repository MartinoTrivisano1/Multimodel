import argparse
import sys

# Importiamo le funzioni principali dai nostri moduli
from train import run_kfold
from evaluate import main as run_evaluate
from explainability import main as run_explainability


def print_banner():
    print("=" * 60)
    print(" 🧠 OASIS MULTIMODAL ALZHEIMER PIPELINE (K-FOLD ENSEMBLE)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Gestore della Pipeline Multimodale per Alzheimer")

    # Definiamo gli argomenti che l'utente può passare
    parser.add_argument(
        '--step',
        type=str,
        choices=['train', 'eval', 'explain', 'all'],
        required=True,
        help="Scegli quale fase della pipeline eseguire: 'train', 'eval', 'explain', o 'all'."
    )

    args = parser.parse_args()
    print_banner()

    if args.step == 'train' or args.step == 'all':
        print("\n▶ FASE 1: ADDESTRAMENTO K-FOLD")
        run_kfold()

    if args.step == 'eval' or args.step == 'all':
        print("\n▶ FASE 2: VALUTAZIONE ENSEMBLE SUL TEST SET")
        try:
            run_evaluate()
        except Exception as e:
            print(f"\n❌ Errore durante la valutazione: {e}")
            print("Assicurati di aver completato l'addestramento prima di valutare.")
            if args.step == 'all': sys.exit(1)

    if args.step == 'explain' or args.step == 'all':
        print("\n▶ FASE 3: EXPLAINABILITY (Grad-CAM & Feature Importance)")
        try:
            run_explainability()
        except Exception as e:
            print(f"\n❌ Errore durante l'explainability: {e}")
            print("Assicurati di aver completato l'addestramento prima di generare i grafici.")

    print("\n" + "=" * 60)
    print(" ✅ ESECUZIONE COMPLETATA")
    print("=" * 60)


if __name__ == "__main__":
    main()