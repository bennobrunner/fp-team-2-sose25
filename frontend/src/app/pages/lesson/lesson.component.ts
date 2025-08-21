import {Component, inject} from '@angular/core';
import {VideoComponent} from '../../components/video/video.component';
import {RecognizedComponent} from '../../components/recognized/recognized.component';
import {ActualComponent} from '../../components/actual/actual.component';
import {HandLandmarkerResult} from '@mediapipe/tasks-vision';
import {HttpClient} from '@angular/common/http';

@Component({
  selector: 'app-lesson',
  imports: [
    VideoComponent,
    RecognizedComponent,
    ActualComponent
  ],
  templateUrl: './lesson.component.html',
  styleUrl: './lesson.component.scss'
})
export class LessonComponent {

  private http = inject(HttpClient)

  currentLesson = 'E'
  recognizedCharacter = '';

  streak: string[] = []

  checkResult(result: HandLandmarkerResult) {
    const request = result.landmarks.map(landmarks => landmarks.map(landmark => ([landmark.x, landmark.y, landmark.z]))).flat()

    this.http.post<FingerAlphabetResponse>("/api/fingers", {
      landmarks: request
    }).subscribe(res => {
      if (res.character !== "")
      this.recognizedCharacter = res.character

      if (this.recognizedCharacter === this.currentLesson) {
        this.streak.push(this.recognizedCharacter)
        this.nextLesson()
      }
    })
  }

  skipLesson() {
    this.nextLesson()
  }

  private nextLesson() {
    this.currentLesson = String.fromCharCode(65 + Math.floor(Math.random() * 26)) // Random character A-Z
  }
}

interface FingerAlphabetResponse {
  character: string;
}
