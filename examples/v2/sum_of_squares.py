# /// script
# dependencies = [
#    "flyte>=2.0.0b49",
# ]
# ///

import asyncio

import flyte

env = flyte.TaskEnvironment(
    name="sum_of_squares",
    resources=flyte.Resources(cpu=2, memory="512Mi"),
    image=(
        flyte.Image
        .from_uv_script(
            __file__,
            name="flyte",
            registry="ghcr.io/flyteorg",
            platform=("linux/amd64", "linux/arm64"),
            python_version=(3, 13),
            pre=True,
        )
        .with_apt_packages("ca-certificates")
    ),
)


@env.task
async def compute_partial_sum_of_squares(start: int, end: int) -> int:
    """Compute the sum of squares for numbers in the range [start, end)."""
    total = sum(i * i for i in range(start, end))
    print(f"Computed sum of squares for range [{start}, {end}): {total}")
    return total


@env.task
async def sum_partials(partials: list[int]) -> int:
    """Sum all the partial results."""
    total = sum(partials)
    print(f"Total sum of squares: {total}")
    return total


@env.task
async def main(n: int = 1_000_000, chunk_size: int = 10_000) -> int:
    """
    Fan out tasks to compute the square of numbers from 1 to n, then sum the squares.
    
    We partition the work into chunks to avoid creating too many tasks.
    With n=1_000_000 and chunk_size=10_000, we create 100 parallel tasks.
    """
    # Create ranges for each chunk
    ranges = []
    for start in range(1, n + 1, chunk_size):
        end = min(start + chunk_size, n + 1)
        ranges.append((start, end))
    
    print(f"Partitioning {n} numbers into {len(ranges)} chunks of ~{chunk_size} each")
    
    # Fan out: compute partial sums in parallel
    partial_sum_tasks = []
    with flyte.group(f"parallel-sum-of-squares-{len(ranges)}-chunks"):
        for start, end in ranges:
            partial_sum_tasks.append(compute_partial_sum_of_squares(start, end))
        
        partial_sums = await asyncio.gather(*partial_sum_tasks)
    
    # Reduce: sum all partial results
    total = await sum_partials(list(partial_sums))
    
    # Verify with mathematical formula: sum of squares from 1 to n = n(n+1)(2n+1)/6
    expected = n * (n + 1) * (2 * n + 1) // 6
    print(f"Expected (formula): {expected}")
    print(f"Computed: {total}")
    print(f"Match: {total == expected}")
    
    return total


if __name__ == "__main__":
    import argparse
    import os

    from flyte.remote import auth_metadata

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    flyte.init_passthrough(
        project=os.getenv("FLYTE_INTERNAL_EXECUTION_PROJECT"),
        domain=os.getenv("FLYTE_INTERNAL_EXECUTION_DOMAIN"),
    )
    with auth_metadata(("authorization", os.environ["FLYTE_PASSTHROUGH_API_KEY"])):    
        if args.build:
            uri = flyte.build(env.image, wait=False)
            print(f"build run url: {uri}")
        else:
            run = flyte.with_runcontext(mode="remote").run(main)
            print(run.url)
