import UIKit
import AVFoundation
import Vision


//conforms to protocol AVCapturePhotoCaptureDelegate to capture image
class NeutralMouthWidthViewController: UIViewController, AVCapturePhotoCaptureDelegate {
    
    private var captureSession: AVCaptureSession!   //manage flow of data from camera
    private var photoOutput: AVCapturePhotoOutput!  //capture still images
    private var mouthWidths: [CGFloat] = []    //array for mouth widths
    private var timer: Timer?   //timer for taking the pictures for computing avg mouth width i.e. neutral mouth position
    private var captureCount = 0  //count the image captures
    private let captureLimit = 50  //50 images to capture

    
    //button to let user start image capture
    @IBAction func startButtonTapped(_ sender: UIButton) {
        countdownLabel.isHidden = false // Show the countdown label
        countdownLabel.text = "3" // Reset countdown
        startCountdown(from: 3)  // Start the countdown
    }
    
    @IBOutlet weak var countdownLabel: UILabel!
    
    //method for count down
    func startCountdown(from seconds: Int) {
        var countdown = seconds
        countdownLabel.text = "\(countdown)"
        
        // Create a timer to update the countdown
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
            countdown -= 1
            self.countdownLabel.text = "\(countdown)"
            
            if countdown == 0 {
                timer.invalidate() // Stop the timer
                self.countdownLabel.text = "Done!" // Change label to "Done!"
                self.countdownLabel.isHidden = true // Hide countdown label
            }
        }
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()   //setup the camera
        startCaptureTimer()  //start the capture of the image
    }
    
    
    //method to setup the camera to capture the image
    private func setupCamera() {
        
        captureSession = AVCaptureSession()    //session to manage flow of data from the camera to the app (capturing)
        captureSession.sessionPreset = .photo  //preset the session to photo (will be a high quality photo capture)

        //safely get the access to front camera
        guard let frontCamera = AVCaptureDevice.default(for: .video) else {
            fatalError("No front camera found")
        }
        
        //use front camera as device input for the camera session safely
        do {
            let input = try AVCaptureDeviceInput(device: frontCamera)
            captureSession.addInput(input)
        } catch {
            fatalError("Error setting up the camera input: \(error)")
        }

        photoOutput = AVCapturePhotoOutput()  //handle photo capture (output)
        captureSession.addOutput(photoOutput) //add the captured photo to the camera session
        captureSession.startRunning()  //start the camera session
    }

    
    //method to handle timer for capturing the image
    private func startCaptureTimer() {
        
        //timer is 3/50 secs to capture one image
        timer = Timer.scheduledTimer(withTimeInterval: 3.0 / Double(captureLimit), repeats: true) { [weak self] _ in
            self?.capturePhoto()
        }
    }
    
    
    //configure photo settings & capture of photo, delegate them to this view controller
    private func capturePhoto() {
        let settings = AVCapturePhotoSettings()  //to setup the settings for capture
        photoOutput.capturePhoto(with: settings, delegate: self)  //delegate the camera settings & capturing functionality to this view controller
    }

    
    //called when image capture is completed
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        
        //safely unwrap the image data of the captured image
        guard let imageData = photo.fileDataRepresentation(),
              
                //create UIImage from teh image data
                let image = UIImage(data: imageData) else {
            return
        }
        detectMouthWidth(in: image)  //pass the image as parameter to this method to detect width of the mouth
    }

    
    //method to detect the width of the mouth from the capture image
    private func detectMouthWidth(in image: UIImage) {
        
        guard let cgImage = image.cgImage else { return }  //safely unwrap the cgImage property of the UIImage (image)

        //create request to detect face landmarks in the image after processing is finished. Use "weak" point to avoid retain cycle
        let request = VNDetectFaceLandmarksRequest { [weak self] (request, error) in
            
            guard let self = self else { return }  //safely unwrap self ensuring that it's still available within this closure
            
            //see if the results of the request contain face observations. If so, get the face observation from the results
            if let results = request.results as? [VNFaceObservation], let face = results.first {
                
                //check if the face observations contain landmarks and out lips (for mouth)
                if let landmarks = face.landmarks, let mouth = landmarks.outerLips {
                    
                    //Get the first & last points of the outer lips
                    let points = Array(UnsafeBufferPointer(start: mouth.__normalizedPoints, count: mouth.pointCount))
                    
                    //first point = left corner and last point = right corner of the mouth
                    if let leftCorner = points.first, let rightCorner = points.last {
                        
                        //distance between these to extreme points is the width of the mouth in neutral position
                        let width = self.distanceBetween(point1: leftCorner, point2: rightCorner)
                        self.mouthWidths.append(width)  //add the computed width of the mouth to the array
                        self.captureCount += 1 //go to next captured image

                        // stop the capture if it's below the set limit for the capture
                        if self.captureCount >= self.captureLimit {
                            self.timer?.invalidate()  //invalidate the time
                            self.calculateAverageMouthWidth()  //calculate the avg mouth width
                        }
                    }
                }
            }
        }  //completion handler ends

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])  //create the request handler with cgImage & empty options dict (to specify additional configs)
        try? handler.perform([request])  //perform face landmarks request using the above handler
    }

    //method to calcualte the avg width of the mouth from the captured images widths
    private func calculateAverageMouthWidth() {
        let averageWidth = mouthWidths.reduce(0, +) / CGFloat(mouthWidths.count)  //avg width = sum of all widths/count of widths
        print("Average neutral mouth width: \(averageWidth)")
        UserDefaults.standard.set(averageWidth, forKey: "neutralMouthWidth")   // Save the average width as value of key "neutralMouthWidth"
    }

    //compute euclidean distance between leftmost & rightmost points of the mouth
    private func distanceBetween(point1: CGPoint, point2: CGPoint) -> CGFloat {
        return sqrt(pow(point2.x - point1.x, 2) + pow(point2.y - point1.y, 2))
    }
}

