import {Component, inject} from '@angular/core';
import {VideoComponent} from '../../components/video/video.component';
import {RecognizedComponent} from '../../components/recognized/recognized.component';
import {ActualComponent} from '../../components/actual/actual.component';
import {HandLandmarkerResult} from '@mediapipe/tasks-vision';
import {Lesson} from '../../model/letter';
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

  recognizedCharacter = '';

  currentLesson: Lesson = {
    character: {
      character: 'E',
      landmarks: []
    }
  }

  checkResult(result: HandLandmarkerResult) {
    const request = result.landmarks.map(landmarks => landmarks.map(landmark => ([landmark.x, landmark.y, landmark.z]))).flat().flat()

    this.http.post<FingerAlphabetResponse>("/api/fingers", {
      landmarks: request
    }).subscribe(res => {
      this.recognizedCharacter = res.character
    })
  }
}

interface FingerAlphabetResponse {
  character: string;
}
