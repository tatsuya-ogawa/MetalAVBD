//
//  GameViewController.swift
//  MetalAVBD
//
//  Created by Tatsuya Ogawa on 2026/04/07.
//

import UIKit
import MetalKit

// Our iOS specific view controller
class GameViewController: UIViewController, UIGestureRecognizerDelegate {

    var renderer: Renderer!
    var mtkView: MTKView!
    private let sceneButton: UIButton = {
        let button = UIButton(type: .system)
        var config = UIButton.Configuration.plain()
        config.contentInsets = NSDirectionalEdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12)
        config.baseForegroundColor = .white
        button.configuration = config
        button.translatesAutoresizingMaskIntoConstraints = false
        button.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        button.layer.cornerRadius = 8.0
        return button
    }()
    private let projectileModeControl: UISegmentedControl = {
        let control = UISegmentedControl(items: ["Cube", "Sphere", "Torus", "Armadillo"])
        control.selectedSegmentIndex = 0
        control.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        control.selectedSegmentTintColor = UIColor.systemBlue
        control.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .normal)
        control.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .selected)
        control.translatesAutoresizingMaskIntoConstraints = false
        return control
    }()
    private let projectileSizeLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        return label
    }()
    private let projectileSizeSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 0.25
        slider.maximumValue = 20.0
        return slider
    }()
    private let projectileMassLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        return label
    }()
    private let projectileMassSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 0.1
        slider.maximumValue = 25.0
        return slider
    }()
    private let projectileSpeedLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        return label
    }()
    private let projectileSpeedSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 1.0
        slider.maximumValue = 200.0
        return slider
    }()
    private let projectileFrictionLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        return label
    }()
    private let projectileFrictionSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 0.0
        slider.maximumValue = 2.0
        return slider
    }()
    private let torusRenderModeControl: UISegmentedControl = {
        let control = UISegmentedControl(items: ["Proxy", "Torus"])
        control.selectedSegmentIndex = 1
        control.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        control.selectedSegmentTintColor = UIColor.systemBlue
        control.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .normal)
        control.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .selected)
        control.translatesAutoresizingMaskIntoConstraints = false
        return control
    }()
    private let torusApproxCountLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        label.textAlignment = .center
        return label
    }()
    private let torusApproxCountSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 4
        slider.maximumValue = Float(avbdTorusApproxSphereCountMax)
        return slider
    }()
    private let torusApproxScaleLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        label.textAlignment = .center
        return label
    }()
    private let torusApproxScaleSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 0.25
        slider.maximumValue = 4.0
        return slider
    }()
    private let solverModeControl: UISegmentedControl = {
        let control = UISegmentedControl(items: ["GPU", "CPU"])
        control.selectedSegmentIndex = 0
        control.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        control.selectedSegmentTintColor = UIColor.systemBlue
        control.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .normal)
        control.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .selected)
        control.translatesAutoresizingMaskIntoConstraints = false
        return control
    }()
    private let simulationModeControl: UISegmentedControl = {
        let control = UISegmentedControl(items: ["Manual", "Auto"])
        control.selectedSegmentIndex = 0
        control.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        control.selectedSegmentTintColor = UIColor.systemBlue
        control.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .normal)
        control.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .selected)
        control.translatesAutoresizingMaskIntoConstraints = false
        return control
    }()
    private let stepButton: UIButton = {
        let button = UIButton(type: .system)
        var config = UIButton.Configuration.filled()
        config.title = "Step"
        config.baseForegroundColor = .white
        config.baseBackgroundColor = UIColor.systemOrange
        config.contentInsets = NSDirectionalEdgeInsets(top: 8, leading: 14, bottom: 8, trailing: 14)
        button.configuration = config
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    private let resetButton: UIButton = {
        let button = UIButton(type: .system)
        var config = UIButton.Configuration.filled()
        config.title = "Reset"
        config.baseForegroundColor = .white
        config.baseBackgroundColor = UIColor.systemRed
        config.contentInsets = NSDirectionalEdgeInsets(top: 8, leading: 14, bottom: 8, trailing: 14)
        button.configuration = config
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    private let gpuOptionsStackView: UIStackView = {
        let stackView = UIStackView()
        stackView.translatesAutoresizingMaskIntoConstraints = false
        stackView.axis = .vertical
        stackView.alignment = .trailing
        stackView.spacing = 12
        return stackView
    }()
    private let broadphaseRefreshContainerView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        view.layer.cornerRadius = 8.0
        return view
    }()
    private let broadphaseRefreshLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        label.numberOfLines = 2
        return label
    }()
    private let broadphaseRefreshSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = Float(AVBDSceneDefaults.minBroadphaseFullRefreshStepCount)
        slider.maximumValue = Float(AVBDSceneDefaults.maxBroadphaseFullRefreshStepCount)
        return slider
    }()
    private let warmstartContainerView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        view.layer.cornerRadius = 8.0
        return view
    }()
    private let warmstartLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.text = "Warmstart"
        label.textColor = .white
        label.font = .systemFont(ofSize: 13, weight: .medium)
        return label
    }()
    private let warmstartSwitch: UISwitch = {
        let toggle = UISwitch()
        toggle.translatesAutoresizingMaskIntoConstraints = false
        toggle.onTintColor = UIColor.systemOrange
        return toggle
    }()
    private let collisionSDFBoundsContainerView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        view.layer.cornerRadius = 8.0
        return view
    }()
    private let collisionSDFBoundsLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.text = "Show SDF Debug"
        label.textColor = .white
        label.font = .systemFont(ofSize: 13, weight: .medium)
        return label
    }()
    private let collisionSDFBoundsSwitch: UISwitch = {
        let toggle = UISwitch()
        toggle.translatesAutoresizingMaskIntoConstraints = false
        toggle.onTintColor = UIColor.systemRed
        return toggle
    }()
    private let simulationStepDeltaContainerView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        view.layer.cornerRadius = 8.0
        return view
    }()
    private let simulationStepDeltaLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        label.numberOfLines = 2
        return label
    }()
    private let simulationStepDeltaSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 15
        slider.maximumValue = 240
        return slider
    }()
    private let solverIterationContainerView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        view.layer.cornerRadius = 8.0
        return view
    }()
    private let solverIterationLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        return label
    }()
    private let solverIterationSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 1
        slider.maximumValue = 64
        return slider
    }()
    private let simulationStepsPerFrameContainerView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        view.layer.cornerRadius = 8.0
        return view
    }()
    private let simulationStepsPerFrameLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        return label
    }()
    private let simulationStepsPerFrameSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 1
        slider.maximumValue = 16
        return slider
    }()
    private let linearDampingContainerView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        view.layer.cornerRadius = 8.0
        return view
    }()
    private let linearDampingLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        return label
    }()
    private let linearDampingSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 0.0
        slider.maximumValue = 2.0
        return slider
    }()
    private let angularDampingContainerView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        view.layer.cornerRadius = 8.0
        return view
    }()
    private let angularDampingLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textColor = .white
        label.font = .systemFont(ofSize: 12, weight: .medium)
        return label
    }()
    private let angularDampingSlider: UISlider = {
        let slider = UISlider()
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.minimumValue = 0.0
        slider.maximumValue = 4.0
        return slider
    }()
    private let statsContainerView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = UIColor(white: 0.08, alpha: 0.82)
        view.layer.cornerRadius = 10.0
        return view
    }()
    private let statsLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.numberOfLines = 0
        label.textAlignment = .right
        label.textColor = .white
        label.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        return label
    }()

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = view as? MTKView else {
            print("View of Gameview controller is not an MTKView")
            return
        }

        // Select the device to render with.  We choose the default device
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported")
            return
        }
        
#if targetEnvironment(simulator)
        print("Metal 4 is not supported on simulator")
        return
#else
        // Check for Metal 4 support
        if !defaultDevice.supportsFamily(.metal4) {
            print("Metal 4 is not supported")
            return
        }
        
        mtkView.device = defaultDevice
        mtkView.backgroundColor = UIColor.black

        guard let newRenderer = Renderer(metalKitView: mtkView) else {
            print("Renderer cannot be initialized")
            return
        }

        renderer = newRenderer
        renderer.setProjectileKind(.box)
        renderer.onDebugStatsUpdated = { [weak self] stats in
            self?.updateStatsLabel(stats)
        }

        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)

        mtkView.delegate = renderer

        // Use a single tap for projectiles and a drag for orbit so both interactions coexist.
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        tapGesture.numberOfTouchesRequired = 1
        tapGesture.delegate = self
        tapGesture.cancelsTouchesInView = false

        let orbitGesture = UIPanGestureRecognizer(target: self, action: #selector(handleOrbitPan(_:)))
        orbitGesture.minimumNumberOfTouches = 1
        orbitGesture.maximumNumberOfTouches = 1
        orbitGesture.delegate = self
        orbitGesture.cancelsTouchesInView = false
        mtkView.addGestureRecognizer(orbitGesture)

        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        pinchGesture.delegate = self
        pinchGesture.cancelsTouchesInView = false

        mtkView.addGestureRecognizer(tapGesture)
        mtkView.addGestureRecognizer(pinchGesture)

        setupSceneButton()
        setupSolverModeControl()
        setupSimulationModeControl()
        setupStepButton()
        setupResetButton()
        setupGPUOptionControls()
        setupProjectileModeControl()
        setupTorusRenderModeControl()
        setupTorusApproximationControls()
        setupCollisionSDFBoundsControl()
        setupStatsOverlay()
        setupSimulationParameterControls()
        updateGPUOnlyControlsVisibility()
        updateStatsLabel(renderer.currentDebugStats())

#endif
    }

    private func setupSceneButton() {
        view.addSubview(sceneButton)
        sceneButton.addTarget(self, action: #selector(sceneButtonTapped(_:)), for: .touchUpInside)
        updateSceneButtonTitle()

        NSLayoutConstraint.activate([
            sceneButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            sceneButton.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12)
        ])
    }

    private func setupProjectileModeControl() {
        view.addSubview(projectileModeControl)
        view.addSubview(projectileSizeLabel)
        view.addSubview(projectileSizeSlider)
        view.addSubview(projectileMassLabel)
        view.addSubview(projectileMassSlider)
        view.addSubview(projectileSpeedLabel)
        view.addSubview(projectileSpeedSlider)
        view.addSubview(projectileFrictionLabel)
        view.addSubview(projectileFrictionSlider)

        projectileModeControl.selectedSegmentIndex = renderer.currentProjectileKind.rawValue
        projectileModeControl.addTarget(self, action: #selector(projectileModeChanged(_:)), for: .valueChanged)
        projectileSizeSlider.value = renderer.currentProjectileSize
        projectileMassSlider.value = renderer.currentProjectileMass
        projectileSpeedSlider.value = renderer.currentProjectileSpeed
        projectileFrictionSlider.value = renderer.currentProjectileFriction
        projectileSizeSlider.addTarget(self, action: #selector(projectileSizeChanged(_:)), for: .valueChanged)
        projectileMassSlider.addTarget(self, action: #selector(projectileMassChanged(_:)), for: .valueChanged)
        projectileSpeedSlider.addTarget(self, action: #selector(projectileSpeedChanged(_:)), for: .valueChanged)
        projectileFrictionSlider.addTarget(self, action: #selector(projectileFrictionChanged(_:)), for: .valueChanged)
        updateProjectileLabels()

        NSLayoutConstraint.activate([
            projectileModeControl.topAnchor.constraint(equalTo: sceneButton.bottomAnchor, constant: 12),
            projectileModeControl.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            projectileModeControl.widthAnchor.constraint(equalToConstant: 240),

            projectileSizeLabel.topAnchor.constraint(equalTo: projectileModeControl.bottomAnchor, constant: 12),
            projectileSizeLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            projectileSizeSlider.topAnchor.constraint(equalTo: projectileSizeLabel.bottomAnchor, constant: 6),
            projectileSizeSlider.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            projectileSizeSlider.widthAnchor.constraint(equalToConstant: 240),

            projectileMassLabel.topAnchor.constraint(equalTo: projectileSizeSlider.bottomAnchor, constant: 10),
            projectileMassLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            projectileMassSlider.topAnchor.constraint(equalTo: projectileMassLabel.bottomAnchor, constant: 6),
            projectileMassSlider.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            projectileMassSlider.widthAnchor.constraint(equalToConstant: 240),

            projectileSpeedLabel.topAnchor.constraint(equalTo: projectileMassSlider.bottomAnchor, constant: 10),
            projectileSpeedLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            projectileSpeedSlider.topAnchor.constraint(equalTo: projectileSpeedLabel.bottomAnchor, constant: 6),
            projectileSpeedSlider.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            projectileSpeedSlider.widthAnchor.constraint(equalToConstant: 240),

            projectileFrictionLabel.topAnchor.constraint(equalTo: projectileSpeedSlider.bottomAnchor, constant: 10),
            projectileFrictionLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            projectileFrictionSlider.topAnchor.constraint(equalTo: projectileFrictionLabel.bottomAnchor, constant: 6),
            projectileFrictionSlider.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            projectileFrictionSlider.widthAnchor.constraint(equalToConstant: 240)
        ])
    }

    private func setupTorusRenderModeControl() {
        view.addSubview(torusRenderModeControl)
        torusRenderModeControl.selectedSegmentIndex = renderer.currentTorusVisualMode == .solidTorus ? 1 : 0
        torusRenderModeControl.addTarget(self, action: #selector(torusRenderModeChanged(_:)), for: .valueChanged)

        NSLayoutConstraint.activate([
            torusRenderModeControl.topAnchor.constraint(equalTo: projectileFrictionSlider.bottomAnchor, constant: 12),
            torusRenderModeControl.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            torusRenderModeControl.widthAnchor.constraint(equalToConstant: 240)
        ])
    }

    private func setupTorusApproximationControls() {
        view.addSubview(torusApproxCountLabel)
        view.addSubview(torusApproxCountSlider)
        view.addSubview(torusApproxScaleLabel)
        view.addSubview(torusApproxScaleSlider)

        torusApproxCountSlider.value = Float(renderer.currentTorusApproxSphereCount)
        torusApproxScaleSlider.value = renderer.currentTorusApproxSphereRadiusScale
        torusApproxCountSlider.addTarget(self, action: #selector(torusApproximationChanged(_:)), for: .valueChanged)
        torusApproxScaleSlider.addTarget(self, action: #selector(torusApproximationChanged(_:)), for: .valueChanged)
        updateTorusApproximationLabels()

        NSLayoutConstraint.activate([
            torusApproxCountLabel.topAnchor.constraint(equalTo: torusRenderModeControl.bottomAnchor, constant: 12),
            torusApproxCountLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            torusApproxCountSlider.topAnchor.constraint(equalTo: torusApproxCountLabel.bottomAnchor, constant: 6),
            torusApproxCountSlider.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            torusApproxCountSlider.widthAnchor.constraint(equalToConstant: 240),
            torusApproxScaleLabel.topAnchor.constraint(equalTo: torusApproxCountSlider.bottomAnchor, constant: 10),
            torusApproxScaleLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            torusApproxScaleSlider.topAnchor.constraint(equalTo: torusApproxScaleLabel.bottomAnchor, constant: 6),
            torusApproxScaleSlider.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            torusApproxScaleSlider.widthAnchor.constraint(equalToConstant: 240)
        ])
    }

    private func setupSolverModeControl() {
        view.addSubview(solverModeControl)
        solverModeControl.selectedSegmentIndex = renderer.currentSolverMode == .cpu ? 1 : 0
        solverModeControl.addTarget(self, action: #selector(solverModeChanged(_:)), for: .valueChanged)

        NSLayoutConstraint.activate([
            solverModeControl.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            solverModeControl.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12)
        ])
    }

    private func setupSimulationModeControl() {
        view.addSubview(simulationModeControl)
        simulationModeControl.selectedSegmentIndex = renderer.currentSimulationRunMode == .auto ? 1 : 0
        simulationModeControl.addTarget(self, action: #selector(simulationModeChanged(_:)), for: .valueChanged)

        NSLayoutConstraint.activate([
            simulationModeControl.topAnchor.constraint(equalTo: solverModeControl.bottomAnchor, constant: 12),
            simulationModeControl.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12)
        ])
    }

    private func setupStepButton() {
        view.addSubview(stepButton)
        stepButton.addTarget(self, action: #selector(stepButtonTapped(_:)), for: .touchUpInside)
        updateSimulationControls()

        NSLayoutConstraint.activate([
            stepButton.topAnchor.constraint(equalTo: simulationModeControl.bottomAnchor, constant: 12),
            stepButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12)
        ])
    }

    private func setupResetButton() {
        view.addSubview(resetButton)
        resetButton.addTarget(self, action: #selector(resetButtonTapped(_:)), for: .touchUpInside)

        NSLayoutConstraint.activate([
            resetButton.topAnchor.constraint(equalTo: stepButton.bottomAnchor, constant: 12),
            resetButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12)
        ])
    }

    private func setupGPUOptionControls() {
        view.addSubview(gpuOptionsStackView)
        setupBroadphaseRefreshControl()
        setupWarmstartControl()

        gpuOptionsStackView.addArrangedSubview(broadphaseRefreshContainerView)
        gpuOptionsStackView.addArrangedSubview(warmstartContainerView)

        NSLayoutConstraint.activate([
            gpuOptionsStackView.topAnchor.constraint(equalTo: resetButton.bottomAnchor, constant: 12),
            gpuOptionsStackView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12)
        ])
    }

    private func setupBroadphaseRefreshControl() {
        broadphaseRefreshContainerView.addSubview(broadphaseRefreshLabel)
        broadphaseRefreshContainerView.addSubview(broadphaseRefreshSlider)

        broadphaseRefreshSlider.value = Float(renderer.currentBroadphaseFullRefreshStepCount)
        broadphaseRefreshSlider.addTarget(self, action: #selector(broadphaseRefreshChanged(_:)), for: .valueChanged)
        updateBroadphaseRefreshLabel()

        NSLayoutConstraint.activate([
            broadphaseRefreshLabel.topAnchor.constraint(equalTo: broadphaseRefreshContainerView.topAnchor, constant: 8),
            broadphaseRefreshLabel.leadingAnchor.constraint(equalTo: broadphaseRefreshContainerView.leadingAnchor, constant: 12),
            broadphaseRefreshLabel.trailingAnchor.constraint(equalTo: broadphaseRefreshContainerView.trailingAnchor, constant: -12),
            broadphaseRefreshSlider.topAnchor.constraint(equalTo: broadphaseRefreshLabel.bottomAnchor, constant: 6),
            broadphaseRefreshSlider.leadingAnchor.constraint(equalTo: broadphaseRefreshContainerView.leadingAnchor, constant: 12),
            broadphaseRefreshSlider.trailingAnchor.constraint(equalTo: broadphaseRefreshContainerView.trailingAnchor, constant: -12),
            broadphaseRefreshSlider.bottomAnchor.constraint(equalTo: broadphaseRefreshContainerView.bottomAnchor, constant: -8),
            broadphaseRefreshContainerView.widthAnchor.constraint(equalToConstant: 240)
        ])
    }

    private func setupStatsOverlay() {
        view.addSubview(statsContainerView)
        statsContainerView.addSubview(statsLabel)

        NSLayoutConstraint.activate([
            statsContainerView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            statsContainerView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12),
            statsContainerView.leadingAnchor.constraint(greaterThanOrEqualTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 120),

            statsLabel.topAnchor.constraint(equalTo: statsContainerView.topAnchor, constant: 10),
            statsLabel.bottomAnchor.constraint(equalTo: statsContainerView.bottomAnchor, constant: -10),
            statsLabel.leadingAnchor.constraint(equalTo: statsContainerView.leadingAnchor, constant: 12),
            statsLabel.trailingAnchor.constraint(equalTo: statsContainerView.trailingAnchor, constant: -12)
        ])
    }

    private func setupWarmstartControl() {
        let stackView = UIStackView(arrangedSubviews: [warmstartLabel, warmstartSwitch])
        stackView.translatesAutoresizingMaskIntoConstraints = false
        stackView.axis = .horizontal
        stackView.alignment = .center
        stackView.spacing = 10

        warmstartContainerView.addSubview(stackView)

        warmstartSwitch.isOn = renderer.enableContactWarmstart
        warmstartSwitch.addTarget(self, action: #selector(warmstartSwitchChanged(_:)), for: .valueChanged)

        NSLayoutConstraint.activate([
            stackView.topAnchor.constraint(equalTo: warmstartContainerView.topAnchor, constant: 8),
            stackView.bottomAnchor.constraint(equalTo: warmstartContainerView.bottomAnchor, constant: -8),
            stackView.leadingAnchor.constraint(equalTo: warmstartContainerView.leadingAnchor, constant: 12),
            stackView.trailingAnchor.constraint(equalTo: warmstartContainerView.trailingAnchor, constant: -12)
        ])
    }

    private func setupCollisionSDFBoundsControl() {
        view.addSubview(collisionSDFBoundsContainerView)

        let stackView = UIStackView(arrangedSubviews: [collisionSDFBoundsLabel, collisionSDFBoundsSwitch])
        stackView.translatesAutoresizingMaskIntoConstraints = false
        stackView.axis = .horizontal
        stackView.alignment = .center
        stackView.spacing = 10

        collisionSDFBoundsContainerView.addSubview(stackView)

        collisionSDFBoundsSwitch.isOn = renderer.showCollisionMeshSDFBounds
        collisionSDFBoundsSwitch.addTarget(self, action: #selector(collisionSDFBoundsSwitchChanged(_:)), for: .valueChanged)

        NSLayoutConstraint.activate([
            collisionSDFBoundsContainerView.topAnchor.constraint(equalTo: torusApproxScaleSlider.bottomAnchor, constant: 12),
            collisionSDFBoundsContainerView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            collisionSDFBoundsContainerView.widthAnchor.constraint(equalToConstant: 240),
            stackView.topAnchor.constraint(equalTo: collisionSDFBoundsContainerView.topAnchor, constant: 8),
            stackView.bottomAnchor.constraint(equalTo: collisionSDFBoundsContainerView.bottomAnchor, constant: -8),
            stackView.leadingAnchor.constraint(equalTo: collisionSDFBoundsContainerView.leadingAnchor, constant: 12),
            stackView.trailingAnchor.constraint(equalTo: collisionSDFBoundsContainerView.trailingAnchor, constant: -12)
        ])
    }

    private func setupSimulationParameterControls() {
        view.addSubview(simulationStepDeltaContainerView)
        simulationStepDeltaContainerView.addSubview(simulationStepDeltaLabel)
        simulationStepDeltaContainerView.addSubview(simulationStepDeltaSlider)
        simulationStepDeltaSlider.addTarget(self, action: #selector(simulationStepDeltaChanged(_:)), for: .valueChanged)

        view.addSubview(solverIterationContainerView)
        solverIterationContainerView.addSubview(solverIterationLabel)
        solverIterationContainerView.addSubview(solverIterationSlider)
        solverIterationSlider.addTarget(self, action: #selector(solverIterationChanged(_:)), for: .valueChanged)

        view.addSubview(simulationStepsPerFrameContainerView)
        simulationStepsPerFrameContainerView.addSubview(simulationStepsPerFrameLabel)
        simulationStepsPerFrameContainerView.addSubview(simulationStepsPerFrameSlider)
        simulationStepsPerFrameSlider.addTarget(self, action: #selector(simulationStepsPerFrameChanged(_:)), for: .valueChanged)

        view.addSubview(linearDampingContainerView)
        linearDampingContainerView.addSubview(linearDampingLabel)
        linearDampingContainerView.addSubview(linearDampingSlider)
        linearDampingSlider.addTarget(self, action: #selector(linearDampingChanged(_:)), for: .valueChanged)

        view.addSubview(angularDampingContainerView)
        angularDampingContainerView.addSubview(angularDampingLabel)
        angularDampingContainerView.addSubview(angularDampingSlider)
        angularDampingSlider.addTarget(self, action: #selector(angularDampingChanged(_:)), for: .valueChanged)

        NSLayoutConstraint.activate([
            simulationStepDeltaContainerView.topAnchor.constraint(equalTo: gpuOptionsStackView.bottomAnchor, constant: 12),
            simulationStepDeltaContainerView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            simulationStepDeltaContainerView.widthAnchor.constraint(equalToConstant: 240),

            simulationStepDeltaLabel.topAnchor.constraint(equalTo: simulationStepDeltaContainerView.topAnchor, constant: 8),
            simulationStepDeltaLabel.leadingAnchor.constraint(equalTo: simulationStepDeltaContainerView.leadingAnchor, constant: 12),
            simulationStepDeltaLabel.trailingAnchor.constraint(equalTo: simulationStepDeltaContainerView.trailingAnchor, constant: -12),
            simulationStepDeltaSlider.topAnchor.constraint(equalTo: simulationStepDeltaLabel.bottomAnchor, constant: 6),
            simulationStepDeltaSlider.leadingAnchor.constraint(equalTo: simulationStepDeltaContainerView.leadingAnchor, constant: 12),
            simulationStepDeltaSlider.trailingAnchor.constraint(equalTo: simulationStepDeltaContainerView.trailingAnchor, constant: -12),
            simulationStepDeltaSlider.bottomAnchor.constraint(equalTo: simulationStepDeltaContainerView.bottomAnchor, constant: -8),

            solverIterationContainerView.topAnchor.constraint(equalTo: simulationStepDeltaContainerView.bottomAnchor, constant: 12),
            solverIterationContainerView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            solverIterationContainerView.widthAnchor.constraint(equalToConstant: 240),

            solverIterationLabel.topAnchor.constraint(equalTo: solverIterationContainerView.topAnchor, constant: 8),
            solverIterationLabel.leadingAnchor.constraint(equalTo: solverIterationContainerView.leadingAnchor, constant: 12),
            solverIterationLabel.trailingAnchor.constraint(equalTo: solverIterationContainerView.trailingAnchor, constant: -12),
            solverIterationSlider.topAnchor.constraint(equalTo: solverIterationLabel.bottomAnchor, constant: 6),
            solverIterationSlider.leadingAnchor.constraint(equalTo: solverIterationContainerView.leadingAnchor, constant: 12),
            solverIterationSlider.trailingAnchor.constraint(equalTo: solverIterationContainerView.trailingAnchor, constant: -12),
            solverIterationSlider.bottomAnchor.constraint(equalTo: solverIterationContainerView.bottomAnchor, constant: -8),

            simulationStepsPerFrameContainerView.topAnchor.constraint(equalTo: solverIterationContainerView.bottomAnchor, constant: 12),
            simulationStepsPerFrameContainerView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            simulationStepsPerFrameContainerView.widthAnchor.constraint(equalToConstant: 240),

            simulationStepsPerFrameLabel.topAnchor.constraint(equalTo: simulationStepsPerFrameContainerView.topAnchor, constant: 8),
            simulationStepsPerFrameLabel.leadingAnchor.constraint(equalTo: simulationStepsPerFrameContainerView.leadingAnchor, constant: 12),
            simulationStepsPerFrameLabel.trailingAnchor.constraint(equalTo: simulationStepsPerFrameContainerView.trailingAnchor, constant: -12),
            simulationStepsPerFrameSlider.topAnchor.constraint(equalTo: simulationStepsPerFrameLabel.bottomAnchor, constant: 6),
            simulationStepsPerFrameSlider.leadingAnchor.constraint(equalTo: simulationStepsPerFrameContainerView.leadingAnchor, constant: 12),
            simulationStepsPerFrameSlider.trailingAnchor.constraint(equalTo: simulationStepsPerFrameContainerView.trailingAnchor, constant: -12),
            simulationStepsPerFrameSlider.bottomAnchor.constraint(equalTo: simulationStepsPerFrameContainerView.bottomAnchor, constant: -8),

            linearDampingContainerView.topAnchor.constraint(equalTo: simulationStepsPerFrameContainerView.bottomAnchor, constant: 12),
            linearDampingContainerView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            linearDampingContainerView.widthAnchor.constraint(equalToConstant: 240),

            linearDampingLabel.topAnchor.constraint(equalTo: linearDampingContainerView.topAnchor, constant: 8),
            linearDampingLabel.leadingAnchor.constraint(equalTo: linearDampingContainerView.leadingAnchor, constant: 12),
            linearDampingLabel.trailingAnchor.constraint(equalTo: linearDampingContainerView.trailingAnchor, constant: -12),
            linearDampingSlider.topAnchor.constraint(equalTo: linearDampingLabel.bottomAnchor, constant: 6),
            linearDampingSlider.leadingAnchor.constraint(equalTo: linearDampingContainerView.leadingAnchor, constant: 12),
            linearDampingSlider.trailingAnchor.constraint(equalTo: linearDampingContainerView.trailingAnchor, constant: -12),
            linearDampingSlider.bottomAnchor.constraint(equalTo: linearDampingContainerView.bottomAnchor, constant: -8),

            angularDampingContainerView.topAnchor.constraint(equalTo: linearDampingContainerView.bottomAnchor, constant: 12),
            angularDampingContainerView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -12),
            angularDampingContainerView.widthAnchor.constraint(equalToConstant: 240),
            angularDampingContainerView.bottomAnchor.constraint(lessThanOrEqualTo: statsContainerView.topAnchor, constant: -12),

            angularDampingLabel.topAnchor.constraint(equalTo: angularDampingContainerView.topAnchor, constant: 8),
            angularDampingLabel.leadingAnchor.constraint(equalTo: angularDampingContainerView.leadingAnchor, constant: 12),
            angularDampingLabel.trailingAnchor.constraint(equalTo: angularDampingContainerView.trailingAnchor, constant: -12),
            angularDampingSlider.topAnchor.constraint(equalTo: angularDampingLabel.bottomAnchor, constant: 6),
            angularDampingSlider.leadingAnchor.constraint(equalTo: angularDampingContainerView.leadingAnchor, constant: 12),
            angularDampingSlider.trailingAnchor.constraint(equalTo: angularDampingContainerView.trailingAnchor, constant: -12),
            angularDampingSlider.bottomAnchor.constraint(equalTo: angularDampingContainerView.bottomAnchor, constant: -8)
        ])

        updateSimulationParameterControls()
    }

    private func updateStatsLabel(_ stats: RendererDebugStats) {
        let fpsText = stats.fps > 0 ? String(format: "%.1f", stats.fps) : "--"
        let frameText = stats.frameTimeMS > 0 ? String(format: "%.2f ms", stats.frameTimeMS) : "-- ms"
        statsLabel.text = [
            "FPS \(fpsText)",
            "Frame \(frameText)",
            "Step \(stats.stepCount)",
            "Bodies \(stats.bodyCount)",
            "Mode \(stats.solverModeName) / \(stats.simulationModeName)",
            "Broadphase \(stats.broadphaseModeName)",
            "Warmstart \(stats.warmstartModeName)",
            "Collision SDF \(stats.collisionSDFStatusName)"
        ].joined(separator: "\n")
    }

    private func showCollisionSDFBusyState() {
        var stats = renderer.currentDebugStats()
        stats.collisionSDFStatusName = "Building..."
        updateStatsLabel(stats)
        view.setNeedsLayout()
        view.layoutIfNeeded()
    }

    @objc private func projectileModeChanged(_ sender: UISegmentedControl) {
        let kind: AVBDProjectileKind
        switch sender.selectedSegmentIndex {
        case 1:
            kind = .sphere
        case 2:
            kind = .torus
        case 3:
            kind = .armadillo
        default:
            kind = .box
        }
        renderer.setProjectileKind(kind)
    }

    @objc private func projectileSizeChanged(_ sender: UISlider) {
        let size = round(sender.value * 20.0) / 20.0
        sender.value = size
        renderer.setProjectileSize(size)
        updateProjectileLabels()
    }

    @objc private func projectileMassChanged(_ sender: UISlider) {
        let mass = round(sender.value * 10.0) / 10.0
        sender.value = mass
        renderer.setProjectileMass(mass)
        updateProjectileLabels()
    }

    @objc private func projectileSpeedChanged(_ sender: UISlider) {
        let speed = round(sender.value)
        sender.value = speed
        renderer.setProjectileSpeed(speed)
        updateProjectileLabels()
    }

    @objc private func projectileFrictionChanged(_ sender: UISlider) {
        let friction = round(sender.value * 20.0) / 20.0
        sender.value = friction
        renderer.setProjectileFriction(friction)
        updateProjectileLabels()
    }

    @objc private func solverModeChanged(_ sender: UISegmentedControl) {
        let mode: AVBDSolverMode = sender.selectedSegmentIndex == 0 ? .gpu : .cpu
        showCollisionSDFBusyState()
        renderer.setSolverMode(mode)
        sender.selectedSegmentIndex = renderer.currentSolverMode == .cpu ? 1 : 0
        updateGPUOnlyControlsVisibility()
        updateStatsLabel(renderer.currentDebugStats())
    }

    @objc private func torusRenderModeChanged(_ sender: UISegmentedControl) {
        let mode: AVBDTorusVisualMode = sender.selectedSegmentIndex == 0 ? .proxySpheres : .solidTorus
        renderer.setTorusVisualMode(mode)
    }

    @objc private func torusApproximationChanged(_ sender: UIControl) {
        let sphereCount = Int(torusApproxCountSlider.value.rounded())
        let radiusScale = round(torusApproxScaleSlider.value * 20.0) / 20.0
        torusApproxCountSlider.value = Float(sphereCount)
        torusApproxScaleSlider.value = radiusScale
        renderer.setTorusApproximation(sphereCount: sphereCount, radiusScale: radiusScale)
        updateTorusApproximationLabels()
    }

    private func updateTorusApproximationLabels() {
        torusApproxCountLabel.text = "Approx Count: \(renderer.currentTorusApproxSphereCount)"
        torusApproxScaleLabel.text = String(format: "Approx Radius Scale: %.2fx", renderer.currentTorusApproxSphereRadiusScale)
    }

    private func updateBroadphaseRefreshLabel() {
        let steps = renderer.currentBroadphaseFullRefreshStepCount
        if steps <= 0 {
            broadphaseRefreshLabel.text = "BP Refresh: 0 steps\nAlways Full"
        } else {
            broadphaseRefreshLabel.text = "BP Refresh: \(steps) steps\n0 steps = Always Full"
        }
    }

    private func updateProjectileLabels() {
        projectileSizeLabel.text = String(format: "Projectile Size: %.2f", renderer.currentProjectileSize)
        projectileMassLabel.text = String(format: "Projectile Mass: %.1f", renderer.currentProjectileMass)
        projectileSpeedLabel.text = String(format: "Projectile Speed: %.0f", renderer.currentProjectileSpeed)
        projectileFrictionLabel.text = String(format: "Projectile Friction: %.2f", renderer.currentProjectileFriction)
    }

    private func updateSimulationParameterControls() {
        let dt = renderer.currentSimulationStepDeltaTime
        let hz = 1.0 / Double(dt)
        simulationStepDeltaSlider.value = Float(hz)
        solverIterationSlider.value = Float(renderer.currentSolverIterationCount)
        simulationStepsPerFrameSlider.value = Float(renderer.currentSimulationStepsPerFrame)
        linearDampingSlider.value = renderer.currentLinearDamping
        angularDampingSlider.value = renderer.currentAngularDamping
        simulationStepDeltaLabel.text = String(format: "Step dt: %.2f ms (%.0f Hz)", dt * 1000.0, hz.rounded())
        solverIterationLabel.text = "Iterations / step: \(renderer.currentSolverIterationCount)"
        simulationStepsPerFrameLabel.text = "Steps / frame: \(renderer.currentSimulationStepsPerFrame)"
        linearDampingLabel.text = String(format: "Linear Damping: %.2f", renderer.currentLinearDamping)
        angularDampingLabel.text = String(format: "Angular Damping: %.2f", renderer.currentAngularDamping)
    }

    @objc private func simulationModeChanged(_ sender: UISegmentedControl) {
        let mode: AVBDSimulationRunMode = sender.selectedSegmentIndex == 0 ? .manual : .auto
        renderer.setSimulationRunMode(mode)
        updateSimulationControls()
        updateStatsLabel(renderer.currentDebugStats())
    }

    @objc private func simulationStepDeltaChanged(_ sender: UISlider) {
        let hz = Float(sender.value.rounded())
        sender.value = hz
        renderer.setSimulationStepDeltaTime(1.0 / hz)
        updateSimulationParameterControls()
    }

    @objc private func solverIterationChanged(_ sender: UISlider) {
        let iterationCount = Int(sender.value.rounded())
        sender.value = Float(iterationCount)
        renderer.setSolverIterationCount(iterationCount)
        updateSimulationParameterControls()
    }

    @objc private func simulationStepsPerFrameChanged(_ sender: UISlider) {
        let stepsPerFrame = Int(sender.value.rounded())
        sender.value = Float(stepsPerFrame)
        renderer.setSimulationStepsPerFrame(stepsPerFrame)
        updateSimulationParameterControls()
    }

    @objc private func linearDampingChanged(_ sender: UISlider) {
        let damping = round(sender.value * 20.0) / 20.0
        sender.value = damping
        renderer.setLinearDamping(damping)
        updateSimulationParameterControls()
    }

    @objc private func angularDampingChanged(_ sender: UISlider) {
        let damping = round(sender.value * 20.0) / 20.0
        sender.value = damping
        renderer.setAngularDamping(damping)
        updateSimulationParameterControls()
    }

    @objc private func broadphaseRefreshChanged(_ sender: UISlider) {
        let stepCount = Int(sender.value.rounded())
        sender.value = Float(stepCount)
        renderer.setBroadphaseFullRefreshStepCount(stepCount)
        updateBroadphaseRefreshLabel()
        updateStatsLabel(renderer.currentDebugStats())
    }

    @objc private func stepButtonTapped(_ sender: UIButton) {
        renderer.requestManualStep()
    }

    @objc private func resetButtonTapped(_ sender: UIButton) {
        showCollisionSDFBusyState()
        renderer.resetSimulation()
        solverModeControl.selectedSegmentIndex = renderer.currentSolverMode == .cpu ? 1 : 0
        broadphaseRefreshSlider.value = Float(renderer.currentBroadphaseFullRefreshStepCount)
        warmstartSwitch.isOn = renderer.enableContactWarmstart
        collisionSDFBoundsSwitch.isOn = renderer.showCollisionMeshSDFBounds
        updateBroadphaseRefreshLabel()
        updateGPUOnlyControlsVisibility()
        updateProjectileLabels()
        updateSimulationControls()
        updateSimulationParameterControls()
        updateStatsLabel(renderer.currentDebugStats())
    }

    @objc private func warmstartSwitchChanged(_ sender: UISwitch) {
        renderer.setContactWarmstartEnabled(sender.isOn)
        updateStatsLabel(renderer.currentDebugStats())
    }

    @objc private func collisionSDFBoundsSwitchChanged(_ sender: UISwitch) {
        renderer.setShowCollisionMeshSDFBounds(sender.isOn)
        updateStatsLabel(renderer.currentDebugStats())
    }

    private func updateSceneButtonTitle() {
        sceneButton.setTitle("Scene: \(renderer.currentSceneID.displayName)", for: .normal)
    }

    private func updateSimulationControls() {
        let isManual = renderer.currentSimulationRunMode == .manual
        simulationModeControl.selectedSegmentIndex = isManual ? 0 : 1
        stepButton.isEnabled = isManual
        stepButton.alpha = isManual ? 1.0 : 0.5
    }

    private func updateGPUOnlyControlsVisibility() {
        let isGPU = renderer.currentSolverMode == .gpu
        broadphaseRefreshContainerView.isHidden = !isGPU
        warmstartContainerView.isHidden = !isGPU
        collisionSDFBoundsContainerView.isHidden = !isGPU
    }

    @objc private func sceneButtonTapped(_ sender: UIButton) {
        let alert = UIAlertController(title: "Select Scene", message: nil, preferredStyle: .actionSheet)
        for sceneID in AVBDSceneID.allCases {
            let title = sceneID == renderer.currentSceneID ? "✓ \(sceneID.displayName)" : sceneID.displayName
            alert.addAction(UIAlertAction(title: title, style: .default) { [weak self] _ in
                guard let self else { return }
                self.showCollisionSDFBusyState()
                self.renderer.setScene(sceneID)
                self.updateSceneButtonTitle()
                self.solverModeControl.selectedSegmentIndex = self.renderer.currentSolverMode == .cpu ? 1 : 0
                self.broadphaseRefreshSlider.value = Float(self.renderer.currentBroadphaseFullRefreshStepCount)
                self.warmstartSwitch.isOn = self.renderer.enableContactWarmstart
                self.collisionSDFBoundsSwitch.isOn = self.renderer.showCollisionMeshSDFBounds
                self.updateBroadphaseRefreshLabel()
                self.updateGPUOnlyControlsVisibility()
                self.updateProjectileLabels()
                self.updateSimulationParameterControls()
                self.updateStatsLabel(self.renderer.currentDebugStats())
            })
        }
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        if let popover = alert.popoverPresentationController {
            popover.sourceView = sender
            popover.sourceRect = sender.bounds
        }

        present(alert, animated: true)
    }

    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        guard let mtkView = view as? MTKView else { return }
        let location = gesture.location(in: mtkView)
        let viewSize = mtkView.drawableSize
        renderer.throwBody(at: location, viewSize: CGSize(width: viewSize.width / mtkView.contentScaleFactor,
                                                          height: viewSize.height / mtkView.contentScaleFactor))
    }

    @objc private func handleOrbitPan(_ gesture: UIPanGestureRecognizer) {
        let delta = gesture.translation(in: gesture.view)
        renderer.orbitCamera(delta: delta)
        gesture.setTranslation(.zero, in: gesture.view)
    }

    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        renderer.zoomCamera(scaleDelta: gesture.scale)
        gesture.scale = 1.0
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        let isOrbitAndPinch =
            (gestureRecognizer is UIPanGestureRecognizer && otherGestureRecognizer is UIPinchGestureRecognizer) ||
            (gestureRecognizer is UIPinchGestureRecognizer && otherGestureRecognizer is UIPanGestureRecognizer)
        return isOrbitAndPinch
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        var currentView: UIView? = touch.view
        while let view = currentView {
            if view is UIControl {
                return false
            }
            currentView = view.superview
        }
        return true
    }
}
