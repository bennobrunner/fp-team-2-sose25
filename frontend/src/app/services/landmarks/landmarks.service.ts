import {inject, Injectable} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {HandLandmarkerResult} from '@mediapipe/tasks-vision';

@Injectable({
  providedIn: 'root'
})
export class LandmarksService {
  private http = inject(HttpClient)


  getCharacterPrediction(handLandmarkerResult: HandLandmarkerResult) {
    const request = handLandmarkerResult.landmarks.map(landmarks => landmarks.map(landmark => ([landmark.x, landmark.y, landmark.z]))).flat()

    return this.http.post<FingerAlphabetResponse>("api/fingers", {
      landmarks: request,
      handedness: handLandmarkerResult.handedness[0][0].categoryName
    })
  }
}

interface FingerAlphabetResponse {
  character: string;
}

