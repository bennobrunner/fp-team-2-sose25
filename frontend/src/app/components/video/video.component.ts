import {Component, ElementRef, OnInit, output, ViewChild} from '@angular/core';
import { DrawingUtils, FaceLandmarker, FaceLandmarkerResult, FilesetResolver, HandLandmarker, HandLandmarkerResult } from '@mediapipe/tasks-vision'
import { HAND_CONNECTIONS } from '@mediapipe/hands'
import { FACEMESH_CONTOURS } from '@mediapipe/face_mesh'
import {NgClass} from '@angular/common';

@Component({
  selector: 'app-video',
  imports: [
    NgClass
  ],
  templateUrl: './video.component.html',
  styleUrl: './video.component.scss'
})
export class VideoComponent implements OnInit {
  @ViewChild('webcam')
  videoStream!: ElementRef<HTMLVideoElement>;

  @ViewChild('outputCanvas')
  outCanvas!: ElementRef<HTMLCanvasElement>
  canvasContext!: CanvasRenderingContext2D;

  handLandmarkResults  = output<HandLandmarkerResult>();
  faceLandmarkResults = output<FaceLandmarkerResult>();

  webcamRunning = false;
  drawLandmarksOnCamera = false;

  handLandmarker?: HandLandmarker;
  faceLandmarker?: FaceLandmarker

  async ngOnInit(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks(
      // path/to/wasm/root
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );


    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: "GPU"
      },
      runningMode: 'VIDEO',
      numHands: 1
    });
    // this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    //   baseOptions: {
    //     modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
    //     delegate: "GPU"
    //   },
    //   runningMode: 'VIDEO',
    //   numFaces: 1
    // })

    this.canvasContext = this.outCanvas.nativeElement.getContext('2d')!;
  }

  async enableCam() {
    const constraints: MediaStreamConstraints = {
      video: {
        height: { ideal: 720 },
        width: { ideal: 1280 },
        facingMode: "user", // front camera
        aspectRatio: { ideal: 16 / 9 }
      },
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      this.videoStream.nativeElement.srcObject = stream;
      this.videoStream.nativeElement.addEventListener("loadeddata", predictWebcam);
    });

    let lastVideoTime = -1;
    let handResults: HandLandmarkerResult;
    let faceResults: FaceLandmarkerResult | undefined;

    this.webcamRunning = true;

    const predictWebcam = async () => {
      this.outCanvas.nativeElement.style.width = `${this.videoStream.nativeElement.offsetWidth}px`;
      this.outCanvas.nativeElement.style.height = `${this.videoStream.nativeElement.offsetHeight}px`;
      this.outCanvas.nativeElement.width = this.videoStream.nativeElement.offsetWidth;
      this.outCanvas.nativeElement.height = this.videoStream.nativeElement.offsetHeight;

      const startTimeMs = performance.now();
      if (lastVideoTime !== this.videoStream?.nativeElement.currentTime && this.handLandmarker) {
        lastVideoTime = this.videoStream.nativeElement.currentTime;
        handResults = this.handLandmarker.detectForVideo(this.videoStream.nativeElement, startTimeMs);
        // Turning off face landmarks for now
        // faceResults = this.faceLandmarker.detectForVideo(this.videoStream.nativeElement, startTimeMs);
      }
      // this.canvasContext.save();
      this.canvasContext.clearRect(0, 0, this.outCanvas.nativeElement.width, this.outCanvas.nativeElement.height);

      const drawingUtils = new DrawingUtils(this.canvasContext)
      const handConnections = HAND_CONNECTIONS.map((val: number[]) => { return { start: val[0], end: val[1] } })
      const faceConnections = FACEMESH_CONTOURS.map((val: number[]) => { return { start: val[0], end: val[1] } })

      if (handResults.landmarks) {
        this.handLandmarkResults.emit(handResults);
        if (this.drawLandmarksOnCamera) {
          for (const landmarks of handResults.landmarks) {
            drawingUtils.drawConnectors(landmarks, handConnections, {
              color: "#00FF00",
              lineWidth: 5
            });
            drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
          }
        }
      }
      if (faceResults?.faceLandmarks) {
        this.faceLandmarkResults.emit(faceResults);
        if (this.drawLandmarksOnCamera) {
          for (const landmarks of faceResults.faceLandmarks) {
            drawingUtils.drawConnectors(landmarks, faceConnections, {
              color: "#00FF00",
              lineWidth: 0.5
            })
            drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 0.1 });
          }
        }
      }

      // this.canvasContext?.restore();
      window.requestAnimationFrame(predictWebcam);
    }
  }
}

