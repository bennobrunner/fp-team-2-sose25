import {Component, input, output} from '@angular/core';

@Component({
  selector: 'app-actual',
  imports: [],
  templateUrl: './actual.component.html',
  styleUrl: './actual.component.scss'
})
export class ActualComponent {
  letterToSign = input<string>()
  skip = output()

  skipLesson() {
    this.skip.emit()
  }
}
