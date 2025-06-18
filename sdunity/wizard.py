import gradio as gr

from . import bootcamp


def create_training_wizard() -> gr.Blocks:
    """Return a simple three-step LoRA training wizard."""
    with gr.Blocks() as demo:
        project_state = gr.State()
        step_state = gr.State(1)
        progress = gr.Markdown("### Step 1 of 3", elem_id="wizard_progress")
        gr.HTML(
            """
            <script>
            function wizTab(id){
              const el = document.getElementById(id);
              if(el){ el.click(); }
            }
            </script>
            """,
            visible=False,
        )
        with gr.Tabs() as tabs:
            with gr.TabItem("1. Basic Info", elem_id="wiz_setup_tab"):
                wiz_type = gr.Radio(
                    ["Character", "Style", "Concept", "Effect"],
                    label="LoRA Type",
                    value="Character",
                )
                wiz_name = gr.Textbox(label="Project Name")
                create_btn = gr.Button("Create Project")
                next_btn1 = gr.Button("Next", variant="primary")
                setup_out = gr.Markdown()
            with gr.TabItem("2. Dataset", elem_id="wiz_data_tab"):
                upload_in = gr.File(
                    label="Upload Images or Zip",
                    file_count="directory",
                    file_types=["image", ".zip"],
                )
                upload_btn = gr.Button("Upload")
                autotag_btn = gr.Button("Auto Tag", variant="secondary")
                grid = gr.HTML()
                msg = gr.Markdown()
                back_btn2 = gr.Button("Back")
                next_btn2 = gr.Button("Next", variant="primary")
            with gr.TabItem("3. Train", elem_id="wiz_train_tab"):
                model_sel = gr.Radio(["SD 1.5", "SDXL", "Pony"], label="Model", value="SD 1.5")
                steps_in = gr.Number(label="Steps", value=1000, precision=0)
                lr_in = gr.Number(label="Learning Rate", value=1e-4)
                review_md = gr.Markdown()
                train_btn = gr.Button("Start Training", variant="primary")
                back_btn3 = gr.Button("Back")
                log_md = gr.Markdown()

        def _create_project(lora_type, name):
            if not name:
                return gr.update(), "Project name required", "### Step 1 of 3", 1
            proj = bootcamp.create_project(name, lora_type)
            return proj.name, f"Created project **{name}**", "### Step 2 of 3", 2

        def _next_from_setup(proj_name):
            if not proj_name:
                return "Please create a project first.", "### Step 1 of 3", 1
            return "", "### Step 2 of 3", 2

        def _upload(proj_name, files):
            proj = bootcamp.BootcampProject.load(proj_name)
            if proj is None:
                return "", "Project not found", "### Step 2 of 3", 2
            uploads = []
            if files:
                if isinstance(files, list):
                    uploads = files
                else:
                    uploads = [files]
            count = bootcamp.import_uploads(proj, uploads)
            html = bootcamp.render_tag_grid(proj)
            return html, f"Imported {count} images", "### Step 3 of 3", 3

        def _autotag(proj_name):
            proj = bootcamp.BootcampProject.load(proj_name)
            if proj is None:
                return "", "Project not found"
            bootcamp.auto_tag_dataset(proj)
            html = bootcamp.render_tag_grid(proj)
            return html, "Auto tags generated"

        def _next_from_data(proj_name):
            proj = bootcamp.BootcampProject.load(proj_name)
            if proj is None or not proj.images:
                return "Please upload images first.", "### Step 2 of 3", 2
            return "", "### Step 3 of 3", 3

        def _review(proj_name, model_type):
            proj = bootcamp.BootcampProject.load(proj_name)
            if proj is None:
                return "Project not found", 0, 0.0
            params = bootcamp.suggest_params(proj, model_type)
            info = f"### {proj.name}\nType: {proj.lora_type}\nImages: {len(proj.images)}"
            return info, params["steps"], params["learning_rate"]

        def _train(proj_name, model_type, steps, lr):
            proj = bootcamp.BootcampProject.load(proj_name)
            if proj is None:
                yield "Project not found"
                return
            yield from bootcamp.run_training(proj, model_type, steps, lr)

        create_btn.click(
            _create_project,
            inputs=[wiz_type, wiz_name],
            outputs=[project_state, setup_out, progress, step_state],
            js="(p,msg,prog,step)=>{wizTab('wiz_data_tab');}"
        )
        next_btn1.click(
            _next_from_setup,
            inputs=project_state,
            outputs=[setup_out, progress, step_state],
            js="(msg,prog,step)=>{wizTab('wiz_data_tab');}"
        )
        upload_btn.click(
            _upload,
            inputs=[project_state, upload_in],
            outputs=[grid, msg, progress, step_state],
            js="(grid,msg,prog,step)=>{wizTab('wiz_train_tab');}"
        )
        autotag_btn.click(_autotag, inputs=project_state, outputs=[grid, msg])
        back_btn2.click(lambda: ("### Step 1 of 3", 1), outputs=[progress, step_state], js="wizTab('wiz_setup_tab')")
        next_btn2.click(
            _next_from_data,
            inputs=project_state,
            outputs=[msg, progress, step_state],
            js="(m,p,s)=>{wizTab('wiz_train_tab');}"
        )
        model_sel.change(_review, inputs=[project_state, model_sel], outputs=[review_md, steps_in, lr_in])
        train_btn.click(_train, inputs=[project_state, model_sel, steps_in, lr_in], outputs=log_md)
        back_btn3.click(lambda: ("### Step 2 of 3", 2), outputs=[progress, step_state], js="wizTab('wiz_data_tab')")

    return demo
