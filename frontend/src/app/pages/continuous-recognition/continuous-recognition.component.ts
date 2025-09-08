import {Component, DestroyRef, inject, OnInit} from '@angular/core';
import {VideoComponent} from '../../components/video/video.component';
import {HandLandmarkerResult} from '@mediapipe/tasks-vision';
import {LandmarksService} from '../../services/landmarks/landmarks.service';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {Subject, switchMap, throttle, throttleTime} from 'rxjs';

@Component({
  selector: 'app-continuous-recognition',
  imports: [
    VideoComponent
  ],
  templateUrl: './continuous-recognition.component.html',
  styleUrl: './continuous-recognition.component.scss'
})
export class ContinuousRecognitionComponent implements OnInit {
  private landmarksService = inject(LandmarksService)
  private destroyRef = inject(DestroyRef)
  private detections$ = new Subject<HandLandmarkerResult>()

  recognizedSentence: string[] = []
  currentlyRecognized = ''

  private recognitionDebounce = 0
  private readonly DEBOUNCE_TIME = 5 * 2; // 3 seconds at 4 requests per second

  ngOnInit() {
    this.detections$.pipe(
      takeUntilDestroyed(this.destroyRef),
      throttleTime(200),
      switchMap((res) => this.landmarksService.getCharacterPrediction(res)),
    ).subscribe((result) => {
      if (this.currentlyRecognized === result.character && this.recognitionDebounce > this.DEBOUNCE_TIME) {
        // If DEL is recognized, remove the last character, else add the recognized character to the sentence
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
    });
  }

  onHandResult(event: HandLandmarkerResult) {
    // Add SPACE if no hands are recognised for a certain amount of frames
    if (event.landmarks.length === 0) {
      this.recognitionDebounce += 1;
      if (this.recognitionDebounce > this.DEBOUNCE_TIME) {
        if (this.recognizedSentence[this.recognizedSentence.length -1] === ' ') return;
        this.recognizedSentence.push(' ');
        this.recognitionDebounce = 0;
      }
      return;
    }
    this.detections$.next(event)
  }
}
