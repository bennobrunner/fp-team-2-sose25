import { Component } from '@angular/core';
import {VideoComponent} from '../../components/video/video.component';
import {RecognizedComponent} from '../../components/recognized/recognized.component';
import {ActualComponent} from '../../components/actual/actual.component';
import {HandLandmarkerResult} from '@mediapipe/tasks-vision';
import {Lesson} from '../../model/letter';

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

  currentLesson: Lesson = {
    character: {
      character: 'E',
      landmarks: []
    }
  }

  checkResult(result: HandLandmarkerResult) {
    if (result.landmarks === this.currentLesson.character.landmarks) {

    }
  }
}
