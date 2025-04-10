import os
import time
import gc
import psutil
import torch
import trimesh
import threading

import hou  # Houdini module
from PySide2 import QtWidgets, QtCore, QtGui

# -------------------------------------------------------
# Shared Functions and Model Cache
# -------------------------------------------------------

_model_pipelines = {}
_model_lock = threading.Lock()

def get_pipeline(model_variant, device, model_id):
    """Load or retrieve a cached Hunyuan3D pipeline."""
    key = (model_variant, device, model_id)
    with _model_lock:
        if key not in _model_pipelines:
            try:
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
                _model_pipelines[key] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    model_id,
                    subfolder=model_variant,
                    use_safetensors=True,
                    device=device
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Hunyuan3D model: {str(e)}")
    return _model_pipelines[key]

def unload_models_and_clean_gpu():
    """Unload all cached models and perform cleanup."""
    global _model_pipelines
    with _model_lock:
        _model_pipelines.clear()
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return "Models unloaded and GPU cache cleaned successfully."
    except Exception as e:
        return f"Error cleaning GPU: {str(e)}"

def aggressive_gpu_cleanup():
    """Perform aggressive GPU cleanup to minimize fragmentation and lingering references."""
    torch.cuda.synchronize()
    time.sleep(2)  # Allow any asynchronous operations to complete
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as e:
        print("Aggressive cleanup error:", e)
    # Optional: log a memory summary for tracking
    print("GPU Memory Summary after cleanup:")
    print(torch.cuda.memory_summary())
    
    # Additional cleanup measures
    try:
        # Force synchronization and free cached memory
        torch.cuda.synchronize(device=None)
        # Set all tensors to None for any residual objects
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    obj.detach().cpu()
            except:
                pass
        # Run garbage collection multiple times to ensure freed references
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Extended GPU cleanup error: {e}")

# -------------------------------------------------------
# Worker for Mini Mode (Single Image)
# -------------------------------------------------------

class MiniGenerateWorker(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(str)
    progress_value = QtCore.Signal(int)
    
    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """Generate a 3D model from a single image."""
        (input_path, steps, octree_resolution, num_chunks, seed, guidance,
         device, model_variant, remove_bg, out_format, node_name, output_path) = self.params

        process = psutil.Process(os.getpid())
        cpu_before = process.memory_info().rss
        gpu_before = torch.cuda.memory_allocated() if device == "cuda" else None
        t_start = time.time()

        if not input_path or not os.path.exists(input_path):
            self.progress.emit("Invalid image path.")
            self.finished.emit()
            return

        try:
            from PIL import Image
            input_data = Image.open(input_path).convert("RGB")
            if remove_bg:
                from hy3dgen.rembg import BackgroundRemover
                input_data = BackgroundRemover()(input_data)
            self.progress_value.emit(20)
        except Exception as e:
            self.progress.emit(f"Error processing image: {str(e)}")
            self.finished.emit()
            return

        self.progress.emit("Starting mini model generation...")
        self.progress_value.emit(10)

        try:
            pipeline = get_pipeline(model_variant, device, 'tencent/Hunyuan3D-2mini')
            self.progress_value.emit(40)
            mesh = pipeline(
                image=input_data,
                num_inference_steps=steps,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                generator=torch.manual_seed(seed),
                output_type='trimesh',
                guidance_scale=guidance
            )[0]
            self.progress_value.emit(60)
        except Exception as e:
            self.progress.emit(f"Error during model generation: {str(e)}")
            self.finished.emit()
            return

        t_end = time.time()
        elapsed = t_end - t_start
        cpu_used = (process.memory_info().rss - cpu_before) / (1024 * 1024)
        gpu_used = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if device == "cuda" and gpu_before is not None else "N/A"

        self.progress.emit(f"Mini generation completed in {elapsed:.2f} seconds.")
        self.progress.emit(f"Additional CPU RAM used: {cpu_used:.2f} MB.")
        self.progress.emit(f"Peak GPU memory used: {gpu_used} MB.")

        try:
            trimesh.exchange.export.export_mesh(mesh, output_path)
            self.progress_value.emit(80)
        except Exception as e:
            self.progress.emit(f"Error exporting mesh: {str(e)}")
            self.finished.emit()
            return

        try:
            obj = hou.node("/obj")
            new_geo = obj.createNode("geo", node_name=node_name, run_init_scripts=False)
            file_node = new_geo.createNode("file")
            file_node.parm("file").set(output_path)
            new_geo.layoutChildren()
            self.progress_value.emit(100)
        except Exception as e:
            self.progress.emit(f"Error importing mesh: {str(e)}")
            self.finished.emit()
            return

        self.progress.emit("Mini model generated and imported successfully!")
        # More thorough cleanup after job completion
        if device == "cuda":
            pipeline = None  # Release reference
            mesh = None  # Release reference
            torch.cuda.synchronize()
            aggressive_gpu_cleanup()
        self.finished.emit()

# -------------------------------------------------------
# Worker for MV Mode (Multi-View Images)
# -------------------------------------------------------

class MVGenerateWorker(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(str)
    progress_value = QtCore.Signal(int)
    
    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """Generate a 3D model from three multi-view images."""
        (front_path, left_path, back_path, steps, octree_resolution, num_chunks,
         device, model_variant, output_path, node_name) = self.params

        process = psutil.Process(os.getpid())
        cpu_before = process.memory_info().rss
        t_start = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if not all([front_path, left_path, back_path]) or not all(os.path.exists(p) for p in [front_path, left_path, back_path]):
            self.progress.emit("One or more image paths are invalid.")
            self.finished.emit()
            return

        self.progress.emit("Starting mv model generation...")
        self.progress_value.emit(10)

        try:
            pipeline = get_pipeline(model_variant, device, 'tencent/Hunyuan3D-2mv')
            self.progress_value.emit(40)
            mesh = pipeline(
                image={"front": front_path, "left": left_path, "back": back_path},
                num_inference_steps=steps,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                generator=torch.manual_seed(12345),
                output_type='trimesh'
            )[0]
            self.progress_value.emit(60)
        except Exception as e:
            self.progress.emit(f"Error during mv model generation: {str(e)}")
            self.finished.emit()
            return

        t_end = time.time()
        elapsed = t_end - t_start
        cpu_used = (process.memory_info().rss - cpu_before) / (1024 * 1024)
        self.progress.emit(f"MV generation completed in {elapsed:.2f} seconds.")
        self.progress.emit(f"Additional CPU RAM used: {cpu_used:.2f} MB.")

        try:
            mesh.export(output_path)
            self.progress_value.emit(80)
        except Exception as e:
            self.progress.emit(f"Error exporting mesh: {str(e)}")
            self.finished.emit()
            return

        try:
            obj = hou.node("/obj")
            geo_node = obj.createNode("geo", node_name=node_name, run_init_scripts=False)
            for child in geo_node.children():
                child.destroy()
            file_sop = geo_node.createNode("file")
            file_sop.parm("file").set(output_path)
            file_sop.setDisplayFlag(True)
            file_sop.setRenderFlag(True)
            geo_node.layoutChildren()
            self.progress_value.emit(100)
        except Exception as e:
            self.progress.emit(f"Error importing mesh: {str(e)}")
            self.finished.emit()
            return

        self.progress.emit("MV model generated and imported successfully!")
        # More thorough cleanup after job completion
        if device == "cuda":
            pipeline = None  # Release reference
            mesh = None  # Release reference
            torch.cuda.synchronize()
            aggressive_gpu_cleanup()
        self.finished.emit()

# -------------------------------------------------------
# JobWidget for Image Inputs and Previews
# -------------------------------------------------------

class JobWidget(QtWidgets.QWidget):
    def __init__(self, mode, interface, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.interface = interface  # Reference to Interface instance
        layout = QtWidgets.QVBoxLayout(self)

        if mode == "mini":
            image_layout = QtWidgets.QHBoxLayout()
            self.image_line = QtWidgets.QLineEdit()
            self.image_line.setPlaceholderText("Select image file...")
            image_layout.addWidget(self.image_line)
            browse_btn = QtWidgets.QPushButton("Browse")
            browse_btn.clicked.connect(self.browse_image)
            image_layout.addWidget(browse_btn)
            layout.addLayout(image_layout)
            self.preview_label = QtWidgets.QLabel()
            self.preview_label.setFixedSize(100, 100)
            self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(self.preview_label)
            self.image_line.textChanged.connect(lambda: self.update_preview(self.image_line, self.preview_label))
        else:  # mv mode
            for view, placeholder in [("front", "Front view"), ("left", "Left view"), ("back", "Back view")]:
                line = QtWidgets.QLineEdit()
                line.setPlaceholderText(placeholder)
                setattr(self, f"{view}_line", line)
                btn = QtWidgets.QPushButton("Browse")
                btn.clicked.connect(lambda checked=False, l=line: self.browse_image(l))
                h_layout = QtWidgets.QHBoxLayout()
                h_layout.addWidget(line)
                h_layout.addWidget(btn)
                layout.addLayout(h_layout)
                preview = QtWidgets.QLabel()
                preview.setFixedSize(100, 100)
                preview.setAlignment(QtCore.Qt.AlignCenter)
                setattr(self, f"{view}_preview", preview)
                layout.addWidget(preview)
                line.textChanged.connect(lambda _, l=line, p=preview: self.update_preview(l, p))

        remove_btn = QtWidgets.QPushButton("Remove Job")
        remove_btn.clicked.connect(self.remove_self)
        layout.addWidget(remove_btn)

    def browse_image(self, line_edit=None):
        """Open file dialog to select an image."""
        line_edit = line_edit or self.image_line
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if fname:
            line_edit.setText(fname)

    def update_preview(self, line_edit, preview_label):
        """Update image preview when path changes."""
        path = line_edit.text()
        if os.path.exists(path):
            try:
                pixmap = QtGui.QPixmap(path)
                if not pixmap.isNull():
                    preview_label.setPixmap(pixmap.scaled(100, 100, QtCore.Qt.KeepAspectRatio))
                else:
                    preview_label.setText("Invalid image")
            except Exception:
                preview_label.setText("Error loading image")
        else:
            preview_label.setText("No image")

    def remove_self(self):
        """Remove this job widget from the interface."""
        self.interface.remove_job(self)

# -------------------------------------------------------
# Main Interface
# -------------------------------------------------------

def onCreateInterface():
    class Interface(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.thread = None
            self.worker = None
            self.job_queue = []
            self.current_job_index = 0
            self.setWindowTitle("Hunyuan 3D Generator")  # Added window title
            self.initUI()

        def initUI(self):
            """Initialize the user interface."""
            main_layout = QtWidgets.QVBoxLayout(self)

            # Title and Header
            title_label = QtWidgets.QLabel("Hunyuan 3D Generator")
            title_font = QtGui.QFont()
            title_font.setPointSize(14)
            title_font.setBold(True)
            title_label.setFont(title_font)
            title_label.setAlignment(QtCore.Qt.AlignCenter)
            main_layout.addWidget(title_label)
            
            subtitle_label = QtWidgets.QLabel("Generate 3D models from images")
            subtitle_label.setAlignment(QtCore.Qt.AlignCenter)
            main_layout.addWidget(subtitle_label)
            main_layout.addSpacing(10)

            # Mode selection
            mode_layout = QtWidgets.QHBoxLayout()
            mode_label = QtWidgets.QLabel("Select Mode:")
            self.mode_combo = QtWidgets.QComboBox()
            self.mode_combo.addItems(["mini", "mv"])
            self.mode_combo.currentIndexChanged.connect(self.update_ui)
            mode_layout.addWidget(mode_label)
            mode_layout.addWidget(self.mode_combo)
            main_layout.addLayout(mode_layout)

            # Job list area
            self.job_group = QtWidgets.QGroupBox("Jobs")
            job_layout = QtWidgets.QVBoxLayout(self.job_group)
            self.job_layout = QtWidgets.QVBoxLayout()
            job_layout.addLayout(self.job_layout)
            add_job_btn = QtWidgets.QPushButton("Add Job")
            add_job_btn.clicked.connect(self.add_job)
            job_layout.addWidget(add_job_btn)
            main_layout.addWidget(self.job_group)
            self.job_widgets = []
            self.add_job()

            # Common Parameters
            self.params_group = QtWidgets.QGroupBox("Common Parameters")
            params_layout = QtWidgets.QFormLayout(self.params_group)
            self.steps_spin = QtWidgets.QSpinBox()
            self.steps_spin.setRange(1, 100)
            self.steps_spin.setValue(30)
            params_layout.addRow("Steps:", self.steps_spin)
            self.octree_spin = QtWidgets.QSpinBox()
            self.octree_spin.setRange(16, 1024)
            self.octree_spin.setValue(380)
            params_layout.addRow("Octree Resolution:", self.octree_spin)
            self.chunks_spin = QtWidgets.QSpinBox()
            self.chunks_spin.setRange(1000, 5000000)
            self.chunks_spin.setValue(20000)
            params_layout.addRow("Num Chunks:", self.chunks_spin)
            self.device_combo = QtWidgets.QComboBox()
            self.device_combo.addItems(["cuda", "cpu"])
            params_layout.addRow("Device:", self.device_combo)
            self.variant_combo = QtWidgets.QComboBox()
            params_layout.addRow("Model Variant:", self.variant_combo)
            main_layout.addWidget(self.params_group)

            # Additional Parameters
            self.add_params_group = QtWidgets.QGroupBox("Additional Parameters")
            add_params_layout = QtWidgets.QVBoxLayout(self.add_params_group)
            common_layout = QtWidgets.QHBoxLayout()
            self.output_path_edit = QtWidgets.QLineEdit()
            self.output_path_edit.setPlaceholderText("Enter output file path")
            default_path = os.path.join(hou.expandString("$HIP"), "generated_model.obj")
            self.output_path_edit.setText(default_path)
            common_layout.addWidget(self.output_path_edit)
            self.outpath_browse_btn = QtWidgets.QPushButton("Browse Directory")
            self.outpath_browse_btn.clicked.connect(self.browse_directory)
            common_layout.addWidget(self.outpath_browse_btn)
            add_params_layout.addLayout(common_layout)
            self.mini_add_group = QtWidgets.QGroupBox("Mini Additional Parameters")
            mini_add_layout = QtWidgets.QFormLayout(self.mini_add_group)
            self.remove_bg_checkbox = QtWidgets.QCheckBox()
            self.remove_bg_checkbox.setChecked(True)
            mini_add_layout.addRow("Remove Background:", self.remove_bg_checkbox)
            self.out_format_combo = QtWidgets.QComboBox()
            self.out_format_combo.addItems(["OBJ", "GLB", "STL"])
            mini_add_layout.addRow("Output Format:", self.out_format_combo)
            self.node_name_edit = QtWidgets.QLineEdit()
            self.node_name_edit.setPlaceholderText("Enter Houdini node name")
            self.node_name_edit.setText("hunyuan_model")
            mini_add_layout.addRow("Node Name:", self.node_name_edit)
            add_params_layout.addWidget(self.mini_add_group)
            main_layout.addWidget(self.add_params_group)

            # Buttons and Status
            btn_layout = QtWidgets.QHBoxLayout()
            self.generate_btn = QtWidgets.QPushButton("Generate 3D Model")
            self.generate_btn.clicked.connect(self.on_generate_clicked)
            self.generate_btn.setMinimumHeight(40)
            generate_font = QtGui.QFont()
            generate_font.setBold(True)
            self.generate_btn.setFont(generate_font)
            btn_layout.addWidget(self.generate_btn)
            self.clean_gpu_btn = QtWidgets.QPushButton("Clean GPU & Unload Models")
            self.clean_gpu_btn.clicked.connect(self.on_clean_gpu_clicked)
            btn_layout.addWidget(self.clean_gpu_btn)
            main_layout.addLayout(btn_layout)
            
            # Status area
            status_layout = QtWidgets.QVBoxLayout()
            status_label = QtWidgets.QLabel("Status:")
            status_layout.addWidget(status_label)
            self.status_output = QtWidgets.QTextEdit()
            self.status_output.setReadOnly(True)
            self.status_output.setMinimumHeight(100)
            status_layout.addWidget(self.status_output)
            main_layout.addLayout(status_layout)

            # Progress Bar
            progress_layout = QtWidgets.QVBoxLayout()
            progress_label = QtWidgets.QLabel("Progress:")
            progress_layout.addWidget(progress_label)
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            progress_layout.addWidget(self.progress_bar)
            main_layout.addLayout(progress_layout)

            # Footer
            footer_label = QtWidgets.QLabel("Hunyuan 3D Generator - Powered by Tencent")
            footer_label.setAlignment(QtCore.Qt.AlignCenter)
            main_layout.addWidget(footer_label)

            self.update_ui()

        def update_ui(self):
            """Update UI based on selected mode."""
            mode = self.mode_combo.currentText()
            for job_widget in self.job_widgets[:]:
                job_widget.setParent(None)
                job_widget.deleteLater()
            self.job_widgets.clear()
            self.add_job()
            self.variant_combo.clear()
            if mode == "mini":
                self.variant_combo.addItems([
                    "hunyuan3d-dit-v2-mini-fast",
                    "hunyuan3d-dit-v2-mini-turbo",
                    "hunyuan3d-dit-v2-mini"
                ])
                self.setWindowTitle("Hunyuan 3D Generator - Mini Mode")
            else:
                self.variant_combo.addItems([
                    "hunyuan3d-dit-v2-mv-fast",
                    "hunyuan3d-dit-v2-mv-turbo",
                    "hunyuan3d-dit-v2-mv"
                ])
                self.setWindowTitle("Hunyuan 3D Generator - Multi-View Mode")

        def add_job(self):
            """Add a new job widget."""
            mode = self.mode_combo.currentText()
            job_widget = JobWidget(mode, self, parent=self.job_group)
            self.job_widgets.append(job_widget)
            self.job_layout.addWidget(job_widget)

        def remove_job(self, job_widget):
            """Remove a job widget."""
            if job_widget in self.job_widgets:
                self.job_widgets.remove(job_widget)
                job_widget.setParent(None)
                job_widget.deleteLater()

        def browse_directory(self):
            """Browse for output directory."""
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
            if directory:
                full_path = os.path.join(directory, "generated_model.obj")
                self.output_path_edit.setText(full_path)

        def on_clean_gpu_clicked(self):
            """Handle GPU cleanup button click."""
            self.status_output.append("Starting GPU cleanup...")
            result = unload_models_and_clean_gpu()
            self.status_output.append(result)
            aggressive_gpu_cleanup()
            self.status_output.append("Deep GPU cleanup completed.")
            
            # Display current memory usage
            if self.device_combo.currentText() == "cuda":
                try:
                    current_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.status_output.append(f"Current GPU memory usage: {current_mem:.2f} MB")
                except Exception as e:
                    self.status_output.append(f"Could not get memory info: {e}")

        def ensure_output_directory_exists(self, output_path):
            """Ensure the output directory exists."""
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    self.status_output.append(f"Created output directory: {output_dir}")
                except Exception as e:
                    self.status_output.append(f"Failed to create output directory: {str(e)}")
                    return False
            return True

        def on_generate_clicked(self):
            """Handle generate button click with batch processing."""
            # Initial GPU cleanup before starting generation
            self.status_output.append("Preparing for generation - cleaning GPU...")
            result = unload_models_and_clean_gpu()
            self.status_output.append(f"GPU Cleanup: {result}")
            if self.device_combo.currentText() == "cuda":
                torch.cuda.synchronize()
                time.sleep(2)
                aggressive_gpu_cleanup()
                self.status_output.append("Deep GPU cleanup completed.")

            mode = self.mode_combo.currentText()
            total_jobs = len(self.job_widgets)
            if total_jobs == 0:
                self.status_output.append("No jobs to process.")
                return

            steps = self.steps_spin.value()
            octree_resolution = self.octree_spin.value()
            num_chunks = self.chunks_spin.value()
            device = self.device_combo.currentText()
            model_variant = self.variant_combo.currentText()
            base_output_path = self.output_path_edit.text().strip()
            base, ext = os.path.splitext(base_output_path)
            remove_bg = self.remove_bg_checkbox.isChecked()
            out_format = self.out_format_combo.currentText()
            base_node_name = self.node_name_edit.text().strip() or "hunyuan_model"

            # Ensure output directory exists
            if not self.ensure_output_directory_exists(base_output_path):
                return

            self.job_queue = []
            for i, job_widget in enumerate(self.job_widgets, start=1):
                output_path = f"{base}_job{i}{ext}"
                node_name = f"{base_node_name}_job{i}"
                if mode == "mini":
                    input_path = job_widget.image_line.text()
                    if not input_path or not os.path.exists(input_path):
                        self.status_output.append(f"Job {i}: Invalid input image path")
                        continue
                    params = (input_path, steps, octree_resolution, num_chunks, 12345, 7.5,
                              device, model_variant, remove_bg, out_format, node_name, output_path)
                else:
                    front_path = job_widget.front_line.text()
                    left_path = job_widget.left_line.text()
                    back_path = job_widget.back_line.text()
                    if not all([front_path, left_path, back_path]) or not all(os.path.exists(p) for p in [front_path, left_path, back_path]):
                        self.status_output.append(f"Job {i}: One or more image paths are invalid")
                        continue
                    params = (front_path, left_path, back_path, steps, octree_resolution,
                              num_chunks, device, model_variant, output_path, node_name)
                self.job_queue.append(params)

            if not self.job_queue:
                self.status_output.append("No valid jobs to process.")
                return

            self.generate_btn.setEnabled(False)
            self.current_job_index = 1
            self.status_output.append(f"Starting job 1 of {len(self.job_queue)}")
            self.progress_bar.setValue(0)
            params = self.job_queue.pop(0)
            self.start_job(params)

        def start_job(self, params):
            """Start processing a single job."""
            mode = self.mode_combo.currentText()
            if mode == "mini":
                self.worker = MiniGenerateWorker(params)
            else:
                self.worker = MVGenerateWorker(params)
            self.thread = QtCore.QThread()
            self.worker.moveToThread(self.thread)
            self.worker.progress.connect(lambda msg: self.status_output.append(f"Job {self.current_job_index}: {msg}"))
            self.worker.progress_value.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self.on_job_finished)
            self.thread.started.connect(self.worker.run)
            self.thread.start()

        def on_job_finished(self):
            """Handle job completion and perform cleanup before starting next job."""
            self.thread.quit()
            self.thread.wait()
            
            # Thorough GPU cleanup between jobs
            if self.device_combo.currentText() == "cuda":
                self.status_output.append(f"Job {self.current_job_index} completed. Cleaning GPU...")
                aggressive_gpu_cleanup()
                try:
                    current_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                    self.status_output.append(f"Current GPU memory after cleanup: {current_mem:.2f} MB")
                except Exception as e:
                    self.status_output.append(f"Could not get memory info: {e}")
            
            total_jobs = len(self.job_widgets)
            if self.job_queue:
                self.current_job_index += 1
                self.status_output.append(f"Starting job {self.current_job_index} of {total_jobs}")
                self.progress_bar.setValue(0)
                params = self.job_queue.pop(0)
                # Add a short delay to ensure complete cleanup
                QtCore.QTimer.singleShot(2000, lambda: self.start_job(params))
            else:
                self.generate_btn.setEnabled(True)
                self.status_output.append("All jobs completed.")
                self.status_output.append("Performing final GPU cleanup...")
                unload_models_and_clean_gpu()
                aggressive_gpu_cleanup()
                self.status_output.append("Final cleanup completed.")

    return Interface()

# -------------------------------------------------------
# Main: For Testing Outside Houdini
# -------------------------------------------------------

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = onCreateInterface()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec_())
