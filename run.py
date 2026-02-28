"""CLI entry point."""

import argparse

from api.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Image → Minecraft Structure Generator")
    parser.add_argument("mesh", help="Path to 3D mesh file (.obj, .glb, .stl)")
    parser.add_argument("--image", help="Reference image for color projection (when mesh has no colors)")
    parser.add_argument("--resolution", type=int, default=0,
                        help="Voxel resolution in blocks (0 = auto-detect based on mesh analysis)")
    parser.add_argument("--depth-scale", type=float, default=0.0,
                        help="Depth compression (0 = auto-detect, 0.5 = half depth, 1.0 = no compression)")
    parser.add_argument("--provider", default="file", choices=["triposr", "trellis2", "file"],
                        help="Image-to-3D provider (default: file)")
    parser.add_argument("--palette", default="default", help="Block palette name (default: default)")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic block mapping")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--format", default="litematic", help="Output format (default: litematic)")
    args = parser.parse_args()

    result = run_pipeline(
        mesh_path=args.mesh,
        reference_image=args.image,
        voxel_resolution=args.resolution,
        depth_scale=args.depth_scale,
        image_to_3d_provider=args.provider,
        block_palette=args.palette,
        use_semantic_mapping=not args.no_semantic,
        output_dir=args.output_dir,
        output_format=args.format,
    )

    print(f"\nOutput: {result.path}")
    print(f"Blocks: {result.block_count}")
    print(f"Dimensions: {result.dimensions[0]}x{result.dimensions[1]}x{result.dimensions[2]}")


if __name__ == "__main__":
    main()
