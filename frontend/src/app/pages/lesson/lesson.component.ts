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

  currentLesson: Lesson = {
    character: {
      character: 'E',
      landmarks: []
    }
  }

  checkResult(result: HandLandmarkerResult) {
    const request = result.landmarks.flat()
    console.log(request);

    this.http.post("/api/fingers", {
      landmarks: request
    }).subscribe()
  }
}
