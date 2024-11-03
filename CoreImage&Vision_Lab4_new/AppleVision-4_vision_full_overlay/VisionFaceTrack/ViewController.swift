
import UIKit
import AVKit
import Vision

class ViewController: UIViewController {
    
    // Main view for showing camera content.
    @IBOutlet weak var previewView: UIView?
    
    // AVCapture variables to hold sequence data
    var session: AVCaptureSession?      //coordinate flow of data from camera to screen
    var previewLayer: AVCaptureVideoPreviewLayer?  //display visual output from capture session
    
    var videoDataOutput: AVCaptureVideoDataOutput?  //process video frames captured by camera
    var videoDataOutputQueue: DispatchQueue?   //handle video frames processing on separate thread avoiding blocking the main thread
    
    var captureDevice: AVCaptureDevice?   //represents capture device like camera
    var captureDeviceResolution: CGSize = CGSize()  //to store resolutation of the camera
    
    // Layer UI for drawing Vision results (display visual overlays on top of camera feed)
    var rootLayer: CALayer?  //base layer
    var detectionOverlayLayer: CALayer?  //contains overlays for detected objects
    var detectedFaceRectangleShapeLayer: CAShapeLayer?   //to draw rectangles around detected faces
    var detectedFaceLandmarksShapeLayer: CAShapeLayer?   //to draw landmarks like eyets, nose, mouth, etc. on detected facess
    
    // Vision requests
    private var detectionRequests: [VNDetectFaceRectanglesRequest]?   //array used to detect face rectangles in video frames
    private var trackingRequests: [VNTrackObjectRequest]?    //to track objects across video frames
    
    lazy var sequenceRequestHandler = VNSequenceRequestHandler()  //used to perform a sequence of vision requests on series of images

    
    
    // prepare vision framework for detecting & tracking faces. Setup drawing layers for visualizing face detection results
    fileprivate func prepareVisionRequest() {
        
        self.trackingRequests = []  //tracking detected faces
        
        // create a detection request that processes an image and returns face features
        // completion handler does not run immediately, it is run after a face is detected
        let faceDetectionRequest:VNDetectFaceRectanglesRequest = VNDetectFaceRectanglesRequest(completionHandler: self.faceDetectionCompletionHandler)
        
        // Save this facial detection request for later processing
        self.detectionRequests = [faceDetectionRequest]
        
        // setup the tracking of a sequence of features from detection (vision requests)
        self.sequenceRequestHandler = VNSequenceRequestHandler()
        
        // setup drawing layers for showing output of face detection
        self.setupVisionDrawingLayers()
    }
    
    
    
    // MARK: UIViewController overrides
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // setup video for high resolution, drop frames when busy, and front camera
        self.session = self.setupAVCaptureSession()
        
        // setup the vision objects for (1) face detection and (2) tracking
        self.prepareVisionRequest()
        
        // start the capture session and get processing a face!
        self.session?.startRunning()
        
        //Lab:
        testingLabel.isHidden = true    //hide the text label until average width is calculated
        smileImageView.isHidden = true  // Initially hide the similing image view
    }
    
    
    // called when app receives a memory warning from the system
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    // Ensure that the interface stays locked in Portrait.
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .portrait
    }
    
    // Ensure that the interface stays locked in Portrait.
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation {
        return .portrait
    }
    
    
    
    // MARK: Performing Vision Requests
    
    // setup face tracking requests based on detected face observations
    func faceDetectionCompletionHandler(request:VNRequest, error: Error?){
        
        // any errors? If yes, show error msg and try to keep going
        if error != nil {
            print("FaceDetection error: \(String(describing: error)).")
        }
        
        // see if we can get any face features, this will fail if no faces detected (ensure that request & its results are of expected type before proceeding)
        // try to save the face observations to a results vector
        guard let faceDetectionRequest = request as? VNDetectFaceRectanglesRequest,
            let results = faceDetectionRequest.results as? [VNFaceObservation] else {
                return
        }
        
        // if initial face is found (i.e. result is not empty), print this msg
        if !results.isEmpty{
            print("Initial Face found... setting up tracking.")
        }
        
        // if we got here, then a face was detected and we have its features savee. The above face detection was the most computational part of what we did
        // the remaining tracking only needs the results vector of face features so we can process it in the main queue (because we will us it to update UI)
        DispatchQueue.main.async {
            
            // Add the face features to the tracking list (results array)
            for observation in results {
                let faceTrackingRequest = VNTrackObjectRequest(detectedObjectObservation: observation)
                
                // the array starts empty, but this will constantly add to it
                // since on the main queue, there are no race conditions, everything is from a single thread
                // once we add this, it kicks off tracking in another function
                self.trackingRequests?.append(faceTrackingRequest)
                
                // NOTE: if the initial face detection is actually not a face, then the app will continually mess up trying to perform tracking
            }
        }
        
    }
    
    
    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate
    /// - Tag: PerformRequests
    // Handle delegate method callback on receiving a sample buffer. This is where we get the pixel buffer from the camera and need to generate the vision requests
    //called when capture output provides a new video frame (output: captured output, sampleBuffer: sample buffer containing video frame data, connection: from which data was received)
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection)
    {
        var requestHandlerOptions: [VNImageOption: AnyObject] = [:]  //empty dict to hold options for vision request handler
        
        // see if camera has any instrinsic transforms on it
        // if it does, add these to the options for requests to the above dict with the given key
        let cameraIntrinsicData = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil)
        if cameraIntrinsicData != nil {
            requestHandlerOptions[VNImageOption.cameraIntrinsics] = cameraIntrinsicData
        }
        
        // check to see if we can get the pixels for processing from sample buffer
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to obtain a CVPixelBuffer for the current output frame.")
            return
        }
        
        // get portrait orientation for UI (current orientation in EXIF format) for image processing
        let exifOrientation = self.exifOrientationForCurrentDeviceOrientation()
        
        // check if trackingRequests array is initialized
        guard let requests = self.trackingRequests else {
            print("Tracking request array not setup, aborting.")
            return
        }

        // check to see if the tracking request is empty (no face currently detected, initial face detection). if it is empty, perform initial face detection
        if requests.isEmpty{
            self.performInitialDetection(pixelBuffer: pixelBuffer, exifOrientation: exifOrientation, requestHandlerOptions: requestHandlerOptions)
            return
        }
        
        // if tracking was not empty, it means we have detected a face very recently so start performing face tracking
        self.performTracking(requests: requests, pixelBuffer: pixelBuffer, exifOrientation: exifOrientation)
        
        
        // if the requests array is not emtpy, check if it remains empty after tracking, print error msg
        if let newTrackingRequests = self.trackingRequests {
            
            if newTrackingRequests.isEmpty {
                // Nothing was high enough confidence to track, just abort.
                print("Face object lost, resetting detection...")
                return
            }
            
            //if we have valud tracking reuests, perform face landmark detection
            self.performLandmarkDetection(newTrackingRequests: newTrackingRequests, pixelBuffer: pixelBuffer, exifOrientation: exifOrientation,
                                          requestHandlerOptions: requestHandlerOptions)
        }
    }
    
    
    // functionality to run the image detection on pixel buffer (takes pixels buffer, image orientation, dict containing vision request handler)
    func performInitialDetection(pixelBuffer:CVPixelBuffer, exifOrientation:CGImagePropertyOrientation, requestHandlerOptions:[VNImageOption: AnyObject]) {
       
        // create request to perfor Vision request on an image
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: exifOrientation, options: requestHandlerOptions)
        
        // Error hanlding to check if the requests array is not nil
        do {
            if let detectRequests = self.detectionRequests{
                try imageRequestHandler.perform(detectRequests)   // try to detect face and add detected faces
            }
        } catch let error as NSError {
            NSLog("Failed to perform FaceRectangleRequest: %@", error)
        }
    }
    
    
    
    // process tracking requests on provided pixel buffer & udpate tracking requests based on the confidence of the detected objects (only take highly confident observations)
    func performTracking(requests:[VNTrackObjectRequest], pixelBuffer:CVPixelBuffer, exifOrientation:CGImagePropertyOrientation)
    {
        // process pixel buffer with tracking requests to perform tracking requests (faster than full face detection)
        do {
            try self.sequenceRequestHandler.perform(requests, on: pixelBuffer, orientation: exifOrientation)
        } catch let error as NSError {
            NSLog("Failed to perform SequenceRequest: %@", error)
        }
  
        var newTrackingRequests = [VNTrackObjectRequest]()  //hold tracking requests for next round of processing
        
        // go over each request
        for trackingRequest in requests {
            
            // if any valid results in the request the grab the first result & assign it to "observation"
            if let results = trackingRequest.results,
               let observation = results[0] as? VNDetectedObjectObservation {
                
                // check if the request is not marked as last frame (is this tracking request of high confidence?)
                // If it is, add it to processing buffer (0.3 is observation's confidence level)
                if !trackingRequest.isLastFrame {
                    if observation.confidence > 0.3 {
                        trackingRequest.inputObservation = observation  //inputObservation is the current observation
                    }
                    
                    // if confidece is below 0.3, the request is marked as last frame to stop further processing
                    else {
                        trackingRequest.isLastFrame = true
                    }
                    newTrackingRequests.append(trackingRequest)  //add the requests to the new array for next round of tracking
                }
            }
        }
        self.trackingRequests = newTrackingRequests //update property to new tracking requests array containing highly confident observations
    }
    
    
    
    //go thru each tracked face request, create landmark detection request for each face, process those requests to detect facial landmarks
    func performLandmarkDetection(newTrackingRequests:[VNTrackObjectRequest], pixelBuffer:CVPixelBuffer, exifOrientation:CGImagePropertyOrientation, requestHandlerOptions:[VNImageOption: AnyObject]) {

        var faceLandmarkRequests = [VNDetectFaceLandmarksRequest]()  //to hold requests for detecting facial landmarks
        
        // Perform landmark detection on tracked faces.
        for trackingRequest in newTrackingRequests {
            
            // create a request for facial landmarks
            let faceLandmarksRequest = VNDetectFaceLandmarksRequest(completionHandler: self.landmarksCompletionHandler)
            
            // if we have results & if first result can be casted to VNDetectedObjectObservation, assign first result to "observation"
            if let trackingResults = trackingRequest.results,
               let observation = trackingResults[0] as? VNDetectedObjectObservation{
        
                let faceObservation = VNFaceObservation(boundingBox: observation.boundingBox)  ///use bounding box of the observation to represent area where face has been detected
                faceLandmarksRequest.inputFaceObservations = [faceObservation]   // indicating the faces to be analyzed for landmarks
                faceLandmarkRequests.append(faceLandmarksRequest)   //add the landmark requests to the array for future processing
                
                // setup for performing landmark detection
                let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: exifOrientation, options: requestHandlerOptions)
                
                do {
                    try imageRequestHandler.perform(faceLandmarkRequests)   // process image data to detect facial landmarks
                } catch let error as NSError {
                    NSLog("Failed to perform FaceLandmarkRequest: %@", error)
                }
            }
        }
    }
    
    
    //Lab
    var mouthWidths: [CGFloat] = [] // Array to store the captured mouth widths
    var captureCount: Int = 0 // Counter for the number of captures
    var neutralMouthWidth: CGFloat = 0.2 // Property to store the computed average mouth width
    var isCapturing: Bool = false // Flag to indicate if capturing is in progress

    @IBOutlet weak var countdownLabel: UILabel!
    
    @IBAction func startCapturingButtonTapped(_ sender: UIButton) {
        startCapturingMouthWidths() // Start capturing mouth widths
    }

    @IBOutlet weak var testingLabel: UILabel!         //text label for instruction
    @IBOutlet weak var smileImageView: UIImageView!   //similing image view
    
    // Lab: Method to capture mouth widths once the button is pressed
    func startCapturingMouthWidths() {
        mouthWidths.removeAll()      // Reset the array for fresh capture
        captureCount = 0             // Reset capture count
        isCapturing = true           // Set capturing flag to true to start capturing
        print("Started capturing mouth widths.")  // Notify that capturing has started
    }

    
    //Lab: Method to capture mouth width
    func captureMouthWidth(mouthWidth: CGFloat) {
        guard isCapturing else { return } // Only capture if capturing is in progress
        mouthWidths.append(mouthWidth)    //add the widths to the array
        captureCount += 1 // Increment the capture count
        print("Captured mouth width: \(mouthWidth)")
        print("Total number of widths: \(mouthWidths.count)")
        
        // If 50 widths have been captured, calculate the average mouth width
        if captureCount >= 50 {
            calculateAverageMouthWidth()
            isCapturing = false // Stop capturing after calculating average
        }
    }
    
    
    //compute euclidean distance between leftmost & rightmost points of the mouth
    private func distanceBetween(point1: CGPoint, point2: CGPoint) -> CGFloat {
        return sqrt(pow(point2.x - point1.x, 2) + pow(point2.y - point1.y, 2))
    }
    
    
    //Lab: Method to calculate the average mouth width
    func calculateAverageMouthWidth() {
        
        //check if there at least one width computed
        guard mouthWidths.count > 0 else {
            print("No mouth widths captured.")
            return
        }
        
        let widthsCount = mouthWidths.count //total number of widths computed
        print("total number of widths: \(widthsCount)")
        
        let averageWidth = mouthWidths.reduce(0, +) / CGFloat(widthsCount)  //computing the avg width of the mouth
        neutralMouthWidth = averageWidth     // Set this as the neutral mouth width
        print("Average Neutral Mouth Width: \(neutralMouthWidth)")
        
        // Show the label of completion of computing the avg width
        DispatchQueue.main.async {
            self.testingLabel.isHidden = false        // Make the label visible
            self.testingLabel.text = "Begin testing!"  // Set the label text
            
            // Hide the label after 3 seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                self.testingLabel.isHidden = true
            }
        }
    }
    
    var isSmiling: Bool = false
    
    //process face landmark detection requests, extract detected landmarks from them, update UI on main thread to draw those landmarks
    func landmarksCompletionHandler(request:VNRequest, error:Error?){
        
        //check if there was error during the request
        if error != nil {
            print("FaceLandmarks error: \(String(describing: error)).")
        }
        
        // any landmarks found that we can display? If not, return
        guard let landmarksRequest = request as? VNDetectFaceLandmarksRequest,
              let results = landmarksRequest.results as? [VNFaceObservation] else {
            return
        }
        
        /*Lab: Start capturing mouth widths only if a face is detected.
        if captureCount == 0 {
            startCapturingMouthWidths()   //Start capturing if it's the first detection
        }
         */
        
        // Perform all UI updates (drawing) on the main queue, not the background queue on which this handler is being called.
        DispatchQueue.main.async {
            
            //iterate thru each face observation to access landmarks
            for faceObservation in results {
                if let landmarks = faceObservation.landmarks{
                    
                    //Lab: Access the mouth landmarks (outer lips)
                    if let mouth = landmarks.outerLips {
                        
                        //extract normalized pts of mouth & conver to array representing key coordinates of mouth
                        let points = Array(UnsafeBufferPointer(start: mouth.__normalizedPoints, count: mouth.pointCount))
                        
                        // Get the left and right corners of the mouth
                        let leftCorner = points.first ?? CGPoint.zero    // the first point is the left corner in the above array
                        let rightCorner = points.last ?? CGPoint.zero    // the last point is the right corner in the above array
                        
                        // Calculate mouth width (distance between left and right corners of the mouth)
                        let mouthWidth = self.distanceBetween(point1: leftCorner, point2: rightCorner)
                        
                        
                        // Check if capturing should start and compute the avg only when "isCapturing" is true (button is pressed)
                        if self.isCapturing {
                            self.captureMouthWidth(mouthWidth: mouthWidth)
                        }
                        
                        // Define smile detection criteria
                        let smileThreshold: CGFloat = 1.1  // Threshold for smiling based on experimentations
                        
                        // Determine if the face is smiling
                        self.isSmiling = mouthWidth > self.neutralMouthWidth * smileThreshold
                        if self.isSmiling {
                            print("The face is smiling.")
                            print("Width while smiling: \(mouthWidth)")
                            self.smileImageView.isHidden = false  // Make smiling image view visible
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                                self.smileImageView.isHidden = true // Hide the image after 0.05 seconds
                            }
                        } else {
                            print("The face is Neutral.")
                        }
                        
                        //Lab: Draw the landmarks using core animation layers
                        self.drawFaceObservations(results)
                        self.updateLandmarkColors(isSmiling: self.isSmiling)
                    }
                } // draw the landmarks using core animation layers (results arry contain detected facial observations)
            }
        }
    }  //end of completion handler
    
}


// MARK: Helper Methods
extension UIViewController{
    
    // Helper Methods for Error Presentation
    //to present alert to user
    fileprivate func presentErrorAlert(withTitle title: String = "Unexpected Failure", message: String) {
        
        //instance of UIAlertController with provided title & msg. preferedStyle: Alert will be presented modally in center of the screen
        let alertController = UIAlertController(title: title, message: message, preferredStyle: .alert)
        self.present(alertController, animated: true)  //present the alert on the screen
    }
    
    
    //to present error alert (detailed info) to the user
    fileprivate func presentError(_ error: NSError) {
        self.presentErrorAlert(withTitle: "Failed with error \(error.code)", message: error.localizedDescription)
    }
    
    
    // Helper Methods for Handling Device Orientation & EXIF
    // Convert input deg to radians
    fileprivate func radiansForDegrees(_ degrees: CGFloat) -> CGFloat {
        return CGFloat(Double(degrees) * Double.pi / 180.0)
    }
    
    
    //convert device orienation to corresponding EXIF orientation ensuring img is orienated correctly based on device orientation
    func exifOrientationForDeviceOrientation(_ deviceOrientation: UIDeviceOrientation) -> CGImagePropertyOrientation {
        
        //handle different cases of device orientation
        switch deviceOrientation {
            
        case .portraitUpsideDown:
            return .rightMirrored   //if portrait upside-down, return right-mirrored
            
        case .landscapeLeft:
            return .downMirrored  //if landscape left, return mirrored down
            
        case .landscapeRight:
            return .upMirrored   //if landscape right, return mirrored up
            
        default:
            return .leftMirrored  //for all other, return left mirrored
        }
    }
    
    
    // get current device orientation to return corresponding EXIF orientation to ensure correct orientation of image
    func exifOrientationForCurrentDeviceOrientation() -> CGImagePropertyOrientation {
        return exifOrientationForDeviceOrientation(UIDevice.current.orientation)
    }
}


// MARK: Extension for AVCapture Setup
extension ViewController:AVCaptureVideoDataOutputSampleBufferDelegate{
    
    /// - Tag: CreateCaptureSession
    // method to setup capture session with front camera, configure video output, designate preview layer & and handle error
    fileprivate func setupAVCaptureSession() -> AVCaptureSession? {
        
        let captureSession = AVCaptureSession()   //to coordinate flow of data from input device to outputs
        
        //handling error
        do {
            let inputDevice = try self.configureFrontCamera(for: captureSession)   //configure front camera for capture session
            
            // configure video data output for capture session using device & resolution obtained from previous step
            self.configureVideoDataOutput(for: inputDevice.device, resolution: inputDevice.resolution, captureSession: captureSession)
            self.designatePreviewLayer(for: captureSession)  //setup preview layer to display video captured by the session
            return captureSession  //return configured captureSession
        } catch let executionError as NSError {
            self.presentError(executionError)   //present error message if the error can be cast to an NSError
        } catch {
            self.presentErrorAlert(message: "An unexpected failure has occured")  //generic error message
        }
        self.teardownAVCapture()  //if error occurs, ensure that capture session is properly cleaned up
        return nil
    }
    

    /// - Tag: ConfigureDeviceResolution
    //find & return highest resolution format (tuple of format & resolution) for a given AVCaptureDevice
    fileprivate func highestResolution420Format(for device: AVCaptureDevice) -> (format: AVCaptureDevice.Format, resolution: CGSize)? {
        
        var highestResolutionFormat: AVCaptureDevice.Format? = nil   //for highest resolution format found
        var highestResolutionDimensions = CMVideoDimensions(width: 0, height: 0) // dims of the highest resolution format found
        
        // iterate thru device formats for given AVCaptureDevice
        for format in device.formats {
            let deviceFormat = format as AVCaptureDevice.Format  //initial format
            let deviceFormatDescription = deviceFormat.formatDescription  //format description of current device format
            
            // if media subtype of the format description is of desired type
            if CMFormatDescriptionGetMediaSubType(deviceFormatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
                let candidateDimensions = CMVideoFormatDescriptionGetDimensions(deviceFormatDescription)  //get dims of current format description
                
                // if the highest resolution format is nil or width of the current format is greater than that of the highest resolution format
                if (highestResolutionFormat == nil) || (candidateDimensions.width > highestResolutionDimensions.width) {
                    highestResolutionFormat = deviceFormat  //set the highest resolution format to current device format
                    highestResolutionDimensions = candidateDimensions  //set highest resolution format dims to that of current device format
                }
            }
        }
        
        //if the highest resolution format is not nil
        if highestResolutionFormat != nil {
            
            //set the object resolution with width & height of the highest resolution format dims
            let resolution = CGSize(width: CGFloat(highestResolutionDimensions.width), height: CGFloat(highestResolutionDimensions.height))
            return (highestResolutionFormat!, resolution)  //return tuple of highest resolution format & its resolution
        }
        
        return nil  // if nothing in the highest resolution format, return nothing
    }
    
    
    //configure front camera with highest resolution format. Return tuple of device & its resolution
    fileprivate func configureFrontCamera(for captureSession: AVCaptureSession) throws -> (device: AVCaptureDevice, resolution: CGSize) {
        
        //create discovery sessionto to find device that supports video for front camera
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .front)
        
        //get the first device from discovery session
        if let device = deviceDiscoverySession.devices.first {
            
            //try to create AVCaptureDeviceInput from the discovered device
            if let deviceInput = try? AVCaptureDeviceInput(device: device) {
                
                // if the captured session can add device input, input is added to the session
                if captureSession.canAddInput(deviceInput) {
                    captureSession.addInput(deviceInput)
                }
                
                //try to get highest resolution format for the device
                if let highestResolution = self.highestResolution420Format(for: device) {
                    try device.lockForConfiguration()   //look at the device config to make changes
                    device.activeFormat = highestResolution.format  //set device active format to the highest resolution format found
                    device.unlockForConfiguration()  //unlock the device config after making changes
                    return (device, highestResolution.resolution)  //return tuple containing configured device and its resolution
                }
            }
        }
        throw NSError(domain: "ViewController", code: 1, userInfo: nil)  //if no suitable device/format found, throw error and code of 1
    }
    
    
    /// - Tag: CreateSerialDispatchQueue
    // Configure video data output for provided AVCaptureSession
    fileprivate func configureVideoDataOutput(for inputDevice: AVCaptureDevice, resolution: CGSize, captureSession: AVCaptureSession) {
        
        //Create video data output to ensure that late video frames are discarded to avoid procesing delays
        let videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        
        // Create a serial dispatch queue used for the sample buffer delegate as well as when a still image is captured.
        // A serial dispatch queue must be used to guarantee that video frames will be processed in order they are received
        let videoDataOutputQueue = DispatchQueue(label: "com.example.apple-samplecode.VisionFaceTrack")
        videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        
        //if the captured session can add the videa data output as an output, the output is added to the captured session
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        }
        
        videoDataOutput.connection(with: .video)?.isEnabled = true  //enable video connection for the video data output
        
        //if the connection is available, check if camera intrinsic matrix delivery is supported. If yes, enable the matrix delivery for connection
        if let captureConnection = videoDataOutput.connection(with: AVMediaType.video) {
            if captureConnection.isCameraIntrinsicMatrixDeliverySupported {
                captureConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
        }
        
        // set these properties of the video data output
        self.videoDataOutput = videoDataOutput
        self.videoDataOutputQueue = videoDataOutputQueue
        
        // set these properties to the input device & resolution
        self.captureDevice = inputDevice
        self.captureDeviceResolution = resolution
    }
    
    
    
    /// - Tag: DesignatePreviewLayer
    // configure & set up video preview layer for given capture session allowing live camera feed to be displayed in UI.
    fileprivate func designatePreviewLayer(for captureSession: AVCaptureSession) {
        
        //Create Vide Preview Layer: using captured session to display live video feed from the session
        let videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        self.previewLayer = videoPreviewLayer
        
        // Set Preview Layer Properties:
        videoPreviewLayer.name = "CameraPreview"  //assign name of the preview layer
        videoPreviewLayer.backgroundColor = UIColor.black.cgColor  //background color of the preview layer to black
        videoPreviewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill  //scale video to fill layer's bounds while preserving video's aspect ratio
        
        //Configure Root layer: if the UIView has a layer
        if let previewRootLayer = self.previewView?.layer {
            self.rootLayer = previewRootLayer  //assign root layer of the view to rootLayer
            previewRootLayer.masksToBounds = true  //set the bounds on root layer ensuring that sublayers are within its bounds
            videoPreviewLayer.frame = previewRootLayer.bounds   //set frame of video preview layer to match bounds of root layer ensuring it cover entire view
            previewRootLayer.addSublayer(videoPreviewLayer)  //add video preview layer as sublayer of root layer to embedd live video feed within view heirarchy
        }
    }
    
    
    // Removes infrastructure for AVCapture as part of cleanup freeing up system resources
    fileprivate func teardownAVCapture() {
       
        //set these properties to nil releasing reference to video data output & associated queue (stop video processing)
        self.videoDataOutput = nil
        self.videoDataOutputQueue = nil
        
        //check if there is valid preview layer. If yes, remove it from superlayer stopping rendering of camera feed in UI
        if let previewLayer = self.previewLayer {
            previewLayer.removeFromSuperlayer()
            self.previewLayer = nil  // release reference to preview layer
        }
    }
}


// MARK: Extension Drawing Vision Observations
extension ViewController {
    
    //configure drawing layers needed for displaying detected faces and their landmarks over camera preview (provide visual feedback to user on detected faces)
    fileprivate func setupVisionDrawingLayers() {
        
        let captureDeviceResolution = self.captureDeviceResolution //get camer's resolution to define size of the drawing layers
        
        //create rectangle (bounds of camera) from origin to dims of device resolution
        let captureDeviceBounds = CGRect(x: 0, y: 0, width: captureDeviceResolution.width, height: captureDeviceResolution.height)
        
        //find center point of device bounds for positioning the layers
        let captureDeviceBoundsCenterPoint = CGPoint(x: captureDeviceBounds.midX, y: captureDeviceBounds.midY)
        
        let normalizedCenterPoint = CGPoint(x: 0.5, y: 0.5)  //center point normalized for layer anchoring purposes
        
        //check if root layer is initialized. If not, present error alert & exit
        guard let rootLayer = self.rootLayer else {
            self.presentErrorAlert(message: "view was not property initialized")
            return
        }
        
        //create overlay layer for drawing detection overlays
        let overlayLayer = CALayer()
        overlayLayer.name = "DetectionOverlay"  //name of the overlay layer
        overlayLayer.masksToBounds = true  //masking
        overlayLayer.anchorPoint = normalizedCenterPoint  //anchor points
        overlayLayer.bounds = captureDeviceBounds  //bounds
        overlayLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY) //position based on device & root layer bounds
        
        //create face rectangle layer
        let faceRectangleShapeLayer = CAShapeLayer()
        faceRectangleShapeLayer.name = "RectangleOutlineLayer"  //name
        faceRectangleShapeLayer.bounds = captureDeviceBounds   //bounds
        faceRectangleShapeLayer.anchorPoint = normalizedCenterPoint  //anchor point
        faceRectangleShapeLayer.position = captureDeviceBoundsCenterPoint  //position
        faceRectangleShapeLayer.fillColor = nil  //no fill color
        faceRectangleShapeLayer.strokeColor = UIColor.green.withAlphaComponent(0.7).cgColor  //stroke color of green
        faceRectangleShapeLayer.lineWidth = 5  //line width
        faceRectangleShapeLayer.shadowOpacity = 0.7   //opacity level
        faceRectangleShapeLayer.shadowRadius = 5  //shadow radius
        
        
        //create face landmarks layer
        let faceLandmarksShapeLayer = CAShapeLayer()
        faceLandmarksShapeLayer.name = "FaceLandmarksLayer"  //name
        faceLandmarksShapeLayer.bounds = captureDeviceBounds  //bounds
        faceLandmarksShapeLayer.anchorPoint = normalizedCenterPoint  //anchor point
        faceLandmarksShapeLayer.position = captureDeviceBoundsCenterPoint   //position
        faceLandmarksShapeLayer.fillColor = nil  //no fill color
        faceLandmarksShapeLayer.strokeColor = UIColor.yellow.withAlphaComponent(0.7).cgColor  //stroke color of yellow
        faceLandmarksShapeLayer.lineWidth = 3  //line width
        faceLandmarksShapeLayer.shadowOpacity = 0.7  //opacity level
        faceLandmarksShapeLayer.shadowRadius = 5  //shadow radius
        
        // Layer Heirarchy
        overlayLayer.addSublayer(faceRectangleShapeLayer)  //add face rectangle layer as sublayer of the overlay layer
        faceRectangleShapeLayer.addSublayer(faceLandmarksShapeLayer)  //add face landmarks layer as sublayer of face rectangle layer
        rootLayer.addSublayer(overlayLayer)  //add overlay layer to root layer integrating it into view heirarchy
        
        //References to created layers are stored
        self.detectionOverlayLayer = overlayLayer
        self.detectedFaceRectangleShapeLayer = faceRectangleShapeLayer
        self.detectedFaceLandmarksShapeLayer = faceLandmarksShapeLayer
        
        self.updateLayerGeometry()  //update layer's layout based on current geometry
    }

    
    
    //adjust pos & transformation of detection overlay layer based on current device orientation & dims of video preview when displaying face detection results
    fileprivate func updateLayerGeometry() {
        
        // check if these three layers are not nil to continue
        guard let overlayLayer = self.detectionOverlayLayer,
            let rootLayer = self.rootLayer,
            let previewLayer = self.previewLayer
            else {
            return
        }
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)  //disable implicit animations for layer changes
        
        //convert specified rectange from metadata output coordiantes (0-1) to layer's coordinates (width & height)
        let videoPreviewRect = previewLayer.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        //variables to hold rotation angle & scale factors for the overlay layer in x & y directions
        var rotation: CGFloat
        var scaleX: CGFloat
        var scaleY: CGFloat
        
        // Rotate the layer into screen orientation.
        switch UIDevice.current.orientation {
        
        //if device is in portrait upside-down, set rotation to 180 deg
        //scale factors for x & y dims based on width & height of video preview rectangle relative to capture device's resolution
        case .portraitUpsideDown:
            rotation = 180
            scaleX = videoPreviewRect.width / captureDeviceResolution.width
            scaleY = videoPreviewRect.height / captureDeviceResolution.height
        
        // if in landscape left orientation, set rotation to 90 deg.
        //scale factor for both dims based on height of video previow rectangle & width of capture device's resolution
        case .landscapeLeft:
            rotation = 90
            scaleX = videoPreviewRect.height / captureDeviceResolution.width
            scaleY = scaleX
        
        // if in landscape right orientation, set rotation to -90 deg. same scale factor as above
        case .landscapeRight:
            rotation = -90
            scaleX = videoPreviewRect.height / captureDeviceResolution.width
            scaleY = scaleX
            
        // for any other orientation, set rotation to 0 deg. Scale factors for x & y as before.
        default:
            rotation = 0
            scaleX = videoPreviewRect.width / captureDeviceResolution.width
            scaleY = videoPreviewRect.height / captureDeviceResolution.height
        }
        
        // Scale and mirror the image to ensure upright presentation
        // affine transformation to combine rotation by specified angle (converted to radians) & scaling in x-direction & inverting y-direction (flip image)
        let affineTransform = CGAffineTransform(rotationAngle: radiansForDegrees(rotation)).scaledBy(x: scaleX, y: -scaleY)
        
        overlayLayer.setAffineTransform(affineTransform) //apply above transformation to overLayer to update pos & orientation to match device orientation & scale factors
        
        // get bounds of rootLayer & set pos of overlayer to center of the rootLayer to cover the entire screen
        let rootLayerBounds = rootLayer.bounds
        overlayLayer.position = CGPoint(x: rootLayerBounds.midX, y: rootLayerBounds.midY)
    }
    

    //add points from landmark region of detected faces to the desired path to tranform them into suitable coordinate space creating a closed shape
    fileprivate func addPoints(in landmarkRegion: VNFaceLandmarkRegion2D, to path: CGMutablePath, applying affineTransform: CGAffineTransform, closingWhenComplete closePath: Bool) {
        
        //get number of points in landmark region
        let pointCount = landmarkRegion.pointCount
        
        // if there are more than one points
        if pointCount > 1 {
            let points: [CGPoint] = landmarkRegion.normalizedPoints  //get normalized landmark points from landmark region (ranges 0-1)
            path.move(to: points[0], transform: affineTransform)  //move first point to adjust it to correct coordinate space
            path.addLines(between: points, transform: affineTransform)  //add lines connecting all points in the pionts array
            
            //if closePath is true, add line back to first point to close the subpath (completes the shape)
            if closePath {
                path.addLine(to: points[0], transform: affineTransform)
                path.closeSubpath() //close subpath to complete the shape
            }
        }
    }
    
    
    //draw rectange around detected face & outline key facial landmarks using separate paths (connecting the dots in the landmarks to draw line)
    fileprivate func addIndicators(to faceRectanglePath: CGMutablePath, faceLandmarksPath: CGMutablePath, for faceObservation: VNFaceObservation) {
        
        let displaySize = self.captureDeviceResolution  //get current resolution of device to calculate size of face rectangle
        
        //bounding box of detected face is converted from normalized to pixel coordinates (defines where face is located in captured image)
        let faceBounds = VNImageRectForNormalizedRect(faceObservation.boundingBox, Int(displaySize.width), Int(displaySize.height))
        faceRectanglePath.addRect(faceBounds)
        
        // check if there are landmarks associated with detected face
        if let landmarks = faceObservation.landmarks {
            
            //map normalized landmark pts into coordinate space of face bounds. Translate pts to origin of face rectangle & scale them to fit rectangle's dims
            let affineTransform = CGAffineTransform(translationX: faceBounds.origin.x, y: faceBounds.origin.y)
                .scaledBy(x: faceBounds.size.width, y: faceBounds.size.height)
            
            // Draw closed landmark regions: Treat eyebrows & lines as open-ended regions when drawing paths. For each region, add pts to the path without closing it
            // end point is not connected to first point
            let openLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEyebrow,
                landmarks.rightEyebrow,
                landmarks.faceContour,
                landmarks.noseCrest,
                landmarks.medianLine
            ]
            for openLandmarkRegion in openLandmarkRegions where openLandmarkRegion != nil {
                self.addPoints(in: openLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: false)
            }
            
            //Draw open landmark regions: Draw eyes, lips, and nose as closed regions. First point is connected end point to close the path
            let closedLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEye,
                landmarks.rightEye,
                landmarks.outerLips,
                landmarks.innerLips,
                landmarks.nose
            ]
            for closedLandmarkRegion in closedLandmarkRegions where closedLandmarkRegion != nil {
                self.addPoints(in: closedLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: true)
            }
        }
    }
    
    
    //Lab: Update the color of the detectedFaceLandmarksShapeLayer based on whether the face is smiling
    fileprivate func updateLandmarkColors(isSmiling: Bool) {
        
        // Set color to green if smiling, otherwise set to yellow
        let color = isSmiling ? UIColor.red.withAlphaComponent(0.7).cgColor : UIColor.yellow.withAlphaComponent(0.7).cgColor
        self.detectedFaceLandmarksShapeLayer?.strokeColor = color
    }
    
    
    // render face rectangles & landmarks on the screen
    fileprivate func drawFaceObservations(_ faceObservations: [VNFaceObservation]) {
        
        // check if the layers used to draw face rectangles & landmarks are nil. If so, exit without performing any drawing
        guard let faceRectangleShapeLayer = self.detectedFaceRectangleShapeLayer,
                let faceLandmarksShapeLayer = self.detectedFaceLandmarksShapeLayer
            else {
            return
        }
        
        CATransaction.begin()  //group mutliple changes to the layer tree so that they can be comitted at once
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)  // disable implicit animations (layer changes immediately without animation)
        
        //two  mutable paths to define shapes for rectangles aroudn detected faces & paths for facial landmarks
        let faceRectanglePath = CGMutablePath()
        let faceLandmarksPath = CGMutablePath()
        
        // add appropriate shapes to each observation based on observed face features
        for faceObservation in faceObservations {
            self.addIndicators(to: faceRectanglePath, faceLandmarksPath: faceLandmarksPath, for: faceObservation)
        }
        
        //assign constructed paths to respective shape layers updating the layers with new shapes that are rendered on the screen
        faceRectangleShapeLayer.path = faceRectanglePath
        faceLandmarksShapeLayer.path = faceLandmarksPath
        
        self.updateLayerGeometry()  //update the geometry of the layers to the current state of the drawing
        CATransaction.commit()  //apply all changes to the layers at once finalizing the drawing operations
    }
       
}





    
