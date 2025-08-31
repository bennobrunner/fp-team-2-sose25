import {Component, inject} from '@angular/core';
import {VideoComponent} from '../../components/video/video.component';
import {HandLandmarkerResult} from '@mediapipe/tasks-vision';
import {LandmarksService} from '../../services/landmarks/landmarks.service';

@Component({
  selector: 'app-continuous-recognition',
  imports: [
    VideoComponent
  ],
  templateUrl: './continuous-recognition.component.html',
  styleUrl: './continuous-recognition.component.scss'
})
export class ContinuousRecognitionComponent {
  private landmarksService = inject(LandmarksService)

  recognizedSentence: string[] = []

   currentlyRecognized = ''

  private recognitionDebounce = 0

  onHandResult(event: HandLandmarkerResult) {
    if (event.landmarks.length === 0) {
      this.recognitionDebounce += 1;
      if (this.recognitionDebounce > 24*5) {
        if (this.recognizedSentence[this.recognizedSentence.length -1] === ' ') return;
        this.recognizedSentence.push(' ');
        this.recognitionDebounce = 0;
      }
      return;
    }

    this.landmarksService.getCharacterPrediction(event).subscribe((result) => {
      if (this.currentlyRecognized === result.character && this.recognitionDebounce > 24*5) {
        if (this.currentlyRecognized === 'DEL') {
          this.recognizedSentence.pop()
        } else {
          this.recognizedSentence.push(result.character)
        }
        this.recognitionDebounce = 0
      }
      else if (this.currentlyRecognized === result.character) {
        this.recognitionDebounce += 1;
      } else if (result.character !== '') {
        this.currentlyRecognized = result.character;
        this.recognitionDebounce = 0;
      }
    })
  }
}
