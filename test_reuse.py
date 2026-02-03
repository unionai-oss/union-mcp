import flyte
import asyncio


env = flyte.TaskEnvironment(
    name="test_reuse",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=flyte.Image.from_debian_base(flyte_version="2.0.0b52").with_pip_packages("unionai-reuse"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        concurrency=1,
        idle_ttl=30,
    ),
)

@env.task
async def test_reuse(x: int) -> int:
    return x * 2

@env.task
async def main(n: int) -> int:
    results = []
    for i in range(n):
        result = test_reuse(i)
        results.append(result)
    results = await asyncio.gather(*results)
    return sum(results)



if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, n=1000)
    print(run.url)
    run.wait()
    print(run.outputs())
