<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"><img src="media/logo.png" width="70">-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction</h1>
  <p align="center">
    <a href="https://www.ifp.uni-stuttgart.de/en/institute/team/Zhang-00004/" target="_blank"><strong>Wei Zhang</strong></a>
    ·
    <a href="https://cvg.cit.tum.de/members/cheq" target="_blank"><strong>Qing Cheng</strong></a>
    ·
    <a href="https://www.ifp.uni-stuttgart.de/en/institute/team/Skuddis/" target="_blank"><strong>David Skuddis</strong></a>
    ·
    <a href="https://www.niclas-zeller.de/" target="_blank"><strong>Niclas Zeller</strong></a>
    ·
    <a href="https://cvg.cit.tum.de/members/cremers" target="_blank"><strong>Daniel Cremers</strong></a>
    ·
    <a href="https://www.ifp.uni-stuttgart.de/en/institute/team/Haala-00001/" target="_blank"><strong>Norbert Haala</strong></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2411.17982">Paper</a> | <a href="https://hi-slam2.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <img src="./media/semantic.gif" width="70%" />
</p>


## Semantic Extension

The Semantic Extension introduces semantic reconstruction capabilities. It enables reconstruct 3D semantic maps utilizing either labeled or predicted 2D semantic maps.

### Running the Semantic Demo

1. **Prepare the Dataset**

   Please download the Replica semantic dataset following the instructions from [SGS-SLAM](https://github.com/ShuhongLL/SGS-SLAM?tab=readme-ov-file#replica). Organize the data in the following structure:
     ```
     data/Replica-semantic/
     └── room0/
         ├── frames/          # RGB images
         ├── semantic_colors/ # Semantic segmentation
     ```

2. **Recompile Gaussian Raserization Kernel**

    The semantic extension requires the updated Gaussian rasterization kernel. Ensure that you recompile the kernel to maintain compatibility:
    ```bash
    pip install thirdparty/diff-gaussian-rasterization 
    ```

3. **Run the Demo**
   ```bash
   python demo.py \
     --imagedir data/Replica-semantic/room0/frames \
     --semanticdir data/Replica-semantic/room0/semantic_colors \
     --config config/replica_config.yaml \
     --calib calib/replica.txt \
     --output outputs/room0 \
     --gsvis
   ```

### Visualization

The demo provides semantic Gaussian Splatting visualization (`--gsvis`). Toggle the "Semantic" button in the rendering options to switch between RGB and semantic rendering modes.
