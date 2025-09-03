// lesson.component.ts
import {Component, DestroyRef, inject, OnInit, OnDestroy} from '@angular/core';
import {Subject, filter, throttleTime, switchMap, takeUntil} from 'rxjs';
import {HandLandmarkerResult} from '@mediapipe/tasks-vision';
import {LandmarksService} from '../../services/landmarks/landmarks.service';

@Component({
  selector: 'app-lesson',
  standalone: true,
  imports: [/* VideoComponent, RecognizedComponent, ActualComponent */],
  templateUrl: './lesson.component.html',
  styleUrl: './lesson.component.scss'
})
export class LessonComponent implements OnInit, OnDestroy {
  private landmarksService = inject(LandmarksService);
  private destroyRef = inject(DestroyRef);

  private detections$ = new Subject<HandLandmarkerResult>();

  currentLesson = this.nextCharacter();
  recognizedCharacter = '';

  // kleines Mehrheitsfenster, damit „ein“ Treffer reicht, ohne lange zu warten
  private lastChars: string[] = [];
  private readonly WINDOW = 6;   // merke die letzten 6 Antworten
  private readonly NEED = 2;     // akzeptiere, wenn mind. 2 davon das Ziel sind

  ngOnInit() {
    this.detections$
      .pipe(
        throttleTime(66, undefined, { trailing: true }), // ~15 FPS
        filter(res => !!res?.landmarks?.length),
        switchMap(res => this.landmarksService.getCharacterPrediction(res)),
        takeUntil(this.destroyRef.onDestroy$)
      )
      .subscribe(res => {
        const ch = res.character ?? '';
        if (ch) this.recognizedCharacter = ch;

        // „schnelle“ Mehrheitslogik, damit nicht auf eine ewig hohe Conf gewartet wird
        this.pushChar(ch);
        if (this.countInWindow(this.currentLesson) >= this.NEED) {
          this.currentLesson = this.nextCharacter();
          this.lastChars.length = 0; // Fenster zurücksetzen
        }
      });
  }

  ngOnDestroy() {}

  // wird pro Video-Frame vom VideoComponent aufgerufen
  checkResult(res: HandLandmarkerResult) {
    this.detections$.next(res);
  }

  skipLesson() {
    this.currentLesson = this.nextCharacter();
    this.lastChars.length = 0;
  }

  private nextCharacter(): string {
    return String.fromCharCode(65 + Math.floor(Math.random() * 26));
  }

  private pushChar(ch: string) {
    if (!ch) return;
    this.lastChars.push(ch);
    if (this.lastChars.length > this.WINDOW) this.lastChars.shift();
  }
  private countInWindow(target: string) {
    return this.lastChars.reduce((n, c) => n + (c === target ? 1 : 0), 0);
  }
}
