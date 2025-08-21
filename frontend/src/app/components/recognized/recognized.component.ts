import {Component, input} from '@angular/core';

@Component({
  selector: 'app-recognized',
  imports: [],
  templateUrl: './recognized.component.html',
  styleUrl: './recognized.component.scss'
})
export class RecognizedComponent {
  letterRecognized = input<string>()
  streak = input<string[]>([])
}
