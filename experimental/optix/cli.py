import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="msmodeling - MindStudio Modeling CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Use 'msmodeling optix -h' for optix-specific help.",
    )
    subparsers = parser.add_subparsers(dest="command", title="commands")

    subparsers.add_parser(
        "optix",
        help="Service Parameter Optimizer for LLM inference performance tuning",
        add_help=False,
    )

    args, remaining = parser.parse_known_args()

    if args.command == "optix":
        from optix.optimizer.optimizer import main as optix_main
        sys.argv = [sys.argv[0]] + remaining
        optix_main()
    else:
        parser.print_help()