import argparse
from src.dataset_setup import get_lobotomized_dataset
from src.trainer import train_model
from src.vector_utils import calculate_and_save_stats
from src.detector import UnknownEntityDetector


def main():
    parser = argparse.ArgumentParser(description="Unknown Entity Detection Project")
    parser.add_argument("--train", action="store_true", help="Phase 1 & 2: Setup data and Train Model")
    parser.add_argument("--stats", action="store_true", help="Phase 3: Calculate 'Safe Zone' Stats (Centroids)")
    parser.add_argument("--predict", type=str, help="Phase 4: Test the detector on a sentence")

    args = parser.parse_args()

    if args.train:
        print(">>> Starting Phase 1: Data Setup...")
        # 1. Get the 'blind' dataset
        dataset = get_lobotomized_dataset()

        print(">>> Starting Phase 2: Training Model...")
        # 2. Train the model
        train_model(dataset)

    elif args.stats:
        print(">>> Starting Phase 3: Computing Class Fingerprints...")
        calculate_and_save_stats()

    elif args.predict:
        print(f">>> Starting Phase 4: Detecting Entities in: '{args.predict}'")
        detector = UnknownEntityDetector()
        results = detector.predict(args.predict)

        print("\n" + "=" * 30)
        print("FINAL RESULTS")
        print("=" * 30)
        for res in results:
            print(f"Word: {res['word']:<15} | Model Thinks: {res['predicted_label']:<10} | Verdict: {res['verdict']}")
            print(
                f"  -> Distance to {res['predicted_label']} cluster: {res['distance']:.4f} (Threshold: {res['threshold']})")
            print("-" * 30)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()