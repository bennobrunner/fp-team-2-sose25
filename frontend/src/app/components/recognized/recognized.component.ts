import {Component, input} from '@angular/core';
import {NgClass} from '@angular/common';

@Component({
  selector: 'app-recognized',
  imports: [NgClass],
  templateUrl: './recognized.component.html',
  styleUrl: './recognized.component.scss',
  standalone: true
})
export class RecognizedComponent {
  letterRecognized = input<string>()
  streak = input<string[]>([]);

}
