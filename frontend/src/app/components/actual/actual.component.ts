import {Component, input, OnChanges, output, SimpleChanges} from '@angular/core';
import {NgClass} from '@angular/common';

@Component({
  selector: 'app-actual',
  imports: [
    NgClass
  ],
  templateUrl: './actual.component.html',
  styleUrl: './actual.component.scss'
})
export class ActualComponent implements OnChanges {
  letterToSign = input<string>()
  skip = output()

  startAnimation = false
  animationTimer: number | undefined = undefined

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['letterToSign']) {
      if (this.animationTimer != 0) {
        clearTimeout(this.animationTimer);
        this.startAnimation = false
      }

      this.startAnimation = true;
      this.animationTimer = setTimeout(() => {
        this.startAnimation = false;
      }, 1000); // Duration of the animation in milliseconds
    }
  }

  skipLesson() {
    this.skip.emit()
  }
}
