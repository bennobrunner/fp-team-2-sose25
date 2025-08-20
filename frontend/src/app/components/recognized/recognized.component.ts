import {Component, input} from '@angular/core';
import {Letter} from '../../model/letter';

@Component({
  selector: 'app-recognized',
  imports: [],
  templateUrl: './recognized.component.html',
  styleUrl: './recognized.component.scss'
})
export class RecognizedComponent {
  letterRecognized = input<string>()

  streak: Letter[] = [{character: "O", landmarks: []}, {character: "O", landmarks: []}, {character: "E", landmarks: []}, {character: "E", landmarks: []}, {character: "A", landmarks: []}, {character: "E", landmarks: []}]

  addToStreak(letter: Letter) {
    this.streak.push(letter);
  }
}
