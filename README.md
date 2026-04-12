# MetalAVBD: Augmented Vertex Block Descent in Metal

![demo](./docs/demo.gif)

[![Metal](https://img.shields.io/badge/Metal-4.0-blue.svg)](https://developer.apple.com/metal/)
[![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg)](https://swift.org/)
[![SIGGRAPH 2025](https://img.shields.io/badge/Research-SIGGRAPH%202025-red.svg)](https://graphics.cs.utah.edu/research/projects/avbd/)
[![Platform](https://img.shields.io/badge/Platform-iOS%20%7C%20macOS-lightgrey.svg)](https://www.apple.com/macos/)

**MetalAVBD** is a GPU-accelerated implementation of the **Augmented Vertex Block Descent (AVBD)** solver in **Metal 4**. 

This project is a Metal-based realization of the research project from the **University of Utah**:
> [**Augmented Vertex Block Descent**](https://graphics.cs.utah.edu/research/projects/avbd/)  
> *Chris Giles, Elie Diaz, and Cem Yuksel*  
> **ACM Transactions on Graphics (Proceedings of SIGGRAPH 2025)**

---

## 📽 Demos & Research

- **[Official Project Page](https://graphics.cs.utah.edu/research/projects/avbd/)** - Research details and comparisons.
- **[Research Paper (PDF)](https://graphics.cs.utah.edu/research/projects/avbd/Augmented_VBD-SIGGRAPH25_RTL.pdf)** - Deep dive into the Augmented Lagrangian formulation.
- **[YouTube Video](https://www.youtube.com/watch?v=bwJgifqvd5M)** - Official demonstration of the solver from the research authors.

<video src="./docs/demo.mp4" width="640" height="auto" controls></video>

---

## 🚀 Key Solver Features

Following the principles of the AVBD research, this implementation offers:

- **Unconditional Stability**: Converges toward the implicit Euler solution, ensuring physical correctness even with large timesteps.
- **Augmented Lagrangian Formulation**: Handles **hard constraints** with infinite stiffness without numerical instabilities or the "softness" common in traditional penalty methods.
- **Convergence**: Improved convergence rates in scenes with high stiffness ratios.
- **GPU Pipeline**: 
  - **Full GPU broadphase** using spatial hashing.
  - **Parallel Gauss-Seidel** solver utilizing contact-aware body coloring for deterministic stability.
- **Scale**: Capable of simulating millions of interacting objects in real-time.

## 🛠 Metal 4 Technical Implementation

This project is optimized for modern Apple Silicon and **Metal 4** features:

- **GPU Task Scheduling**: Leverages adjacency-based coloring to orchestrate constraint solving with minimal contention.
- **Descriptor Management**: Uses **Argument Tables** and **Residency Sets** for efficient resource access.
- **Atomic Memory Management**: Optimized compute shaders using atomic batching to manage dynamic contact manifolds and allocations on the GPU.
- **Diverse Primitive Manifolds**:
  - **Box (SAT)**: Accurate face/edge/vertex collision.
  - **Sphere**: Analytical geometry checks.
  - **Torus**: Advanced collision manifold generation via approximate sphere decomposition.

---

## 📱 Requirements

- **Device**: Physical iOS device (A17 Pro / M1 or later) or Apple Silicon Mac.
- **OS**: iOS 18.0+ / macOS 15.0+ (Metal 4 support required).
- **Tooling**: Xcode 16.0+

> [!IMPORTANT]
> This project utilizes advanced Metal features and **cannot** be run on the iOS Simulator. Please use a physical device or the "macOS Designed for iPad" destination.

## 📜 License & Acknowledgments

This implementation is independent of the original authors' code but is built strictly according to the mathematical foundations laid out in their research.

- **Original Research**: [University of Utah Graphics Lab](https://graphics.cs.utah.edu/)
- **License**: This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
