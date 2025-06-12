import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { DrawingUtils, FaceLandmarker, FaceLandmarkerResult, FilesetResolver, HandLandmarker, HandLandmarkerResult } from '@mediapipe/tasks-vision'
import { HAND_CONNECTIONS } from '@mediapipe/hands'
import { FACEMESH_CONTOURS } from '@mediapipe/face_mesh'

@Component({
  selector: 'app-video',
  imports: [],
  templateUrl: './video.component.html',
  styleUrl: './video.component.scss'
})
export class VideoComponent implements OnInit {
  @ViewChild('webcam')
  videoStream!: ElementRef<HTMLVideoElement>;

  @ViewChild('outputCanvas')
  outCanvas!: ElementRef<HTMLCanvasElement>
  canvasContext!: CanvasRenderingContext2D;

  runningMode = "VIDEO";
  webcamRunning = true;

  handLandmarker?: HandLandmarker;
  faceLandmarker?: FaceLandmarker

  async ngOnInit(): Promise<void> {
    console.log(await navigator.mediaDevices.enumerateDevices());
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
      numHands: 2
    });
    this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
        delegate: "GPU"
      },
      runningMode: 'VIDEO',
      numFaces: 1
    })

    this.canvasContext = this.outCanvas.nativeElement.getContext('2d')!;
  }

  async enableCam() {
    const constraints: MediaStreamConstraints = {
      video: {
        height: 1080,
        width: 1920
      },
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      this.videoStream.nativeElement.srcObject = stream;
      this.videoStream.nativeElement.addEventListener("loadeddata", predictWebcam);
    });

    let lastVideoTime = -1;
    let handResults: HandLandmarkerResult;
    let faceResults: FaceLandmarkerResult;

    const predictWebcam = async () => {
      this.outCanvas.nativeElement.style.width = `${this.videoStream.nativeElement.videoWidth}px`;
      this.outCanvas.nativeElement.style.height = `${this.videoStream.nativeElement.videoHeight}px`;
      this.outCanvas.nativeElement.width = this.videoStream.nativeElement.videoWidth;
      this.outCanvas.nativeElement.height = this.videoStream.nativeElement.videoHeight;

      const startTimeMs = performance.now();
      if (lastVideoTime !== this.videoStream?.nativeElement.currentTime) {
        lastVideoTime = this.videoStream.nativeElement.currentTime;
        handResults = this.handLandmarker!.detectForVideo(this.videoStream.nativeElement, startTimeMs);
        faceResults = this.faceLandmarker!.detectForVideo(this.videoStream.nativeElement, startTimeMs);
      }
      this.canvasContext.save();
      this.canvasContext.clearRect(0, 0, this.outCanvas.nativeElement.width, this.outCanvas.nativeElement.height);

      const drawingUtils = new DrawingUtils(this.canvasContext)
      const handConnections = HAND_CONNECTIONS.map((val: number[]) => { return { start: val[0], end: val[1] } })
      const faceConnections = FACEMESH_CONTOURS.map((val: number[]) => { return { start: val[0], end: val[1] } })


      if (handResults.landmarks) {
        for (const landmarks of handResults.landmarks) {
          console.log(landmarks);
          drawingUtils.drawConnectors(landmarks, handConnections, {
            color: "#00FF00",
            lineWidth: 5
          });
          drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
        }
      }
      if (faceResults.faceLandmarks) {
        for (const landmarks of faceResults.faceLandmarks) {
          drawingUtils.drawConnectors(landmarks, faceConnections, {
            color: "#00FF00",
            lineWidth: 0.5
          })
          drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 0.1 });
        }
      }

      this.canvasContext?.restore();
      window.requestAnimationFrame(predictWebcam);
    }

  }
}

