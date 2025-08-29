import {Component, DestroyRef, inject} from '@angular/core';
import {VideoComponent} from '../../components/video/video.component';
import {RecognizedComponent} from '../../components/recognized/recognized.component';
import {ActualComponent} from '../../components/actual/actual.component';
import {HandLandmarkerResult} from '@mediapipe/tasks-vision';
import {LandmarksService} from '../../services/landmarks/landmarks.service';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';

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
  private landmarksService = inject(LandmarksService)
  private destroyRef = inject(DestroyRef)

  currentLesson = this.nextCharacter()
  recognizedCharacter = '';

  streak: string[] = []

  checkResult(result: HandLandmarkerResult) {
    if (result.landmarks.length <= 0 && result.handedness.length <= 0) {
      return;
    }

    this.landmarksService.getCharacterPrediction(result)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(res => {
        if (res.character !== "")
        this.recognizedCharacter = res.character

        if (this.recognizedCharacter === this.currentLesson) {
          this.streak.push(this.recognizedCharacter)
          this.currentLesson = this.nextCharacter()
        }
    })
  }

  skipLesson() {
    this.currentLesson = this.nextCharacter()
  }

  private nextCharacter(): string {
    return String.fromCharCode(65 + Math.floor(Math.random() * 26)) // Random character A-Z
  }
}
