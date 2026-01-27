import flyte
import flyte.remote


flyte.init_from_config()

script = "print('Hello, World!')"
task = flyte.remote.Task.get(
    name="union_mcp_script_runner.build_script_image_task",
    auto_version="latest",
)
run: flyte.remote.Run = flyte.with_runcontext(mode="remote", copy_style="none", version="1").run(task, script=script)
print(run.url)
run.wait()
print(run.outputs())
