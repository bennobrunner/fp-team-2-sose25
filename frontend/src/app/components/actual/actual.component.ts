import {Component, input} from '@angular/core';
import {Lesson} from '../../model/letter';
import {NgOptimizedImage} from '@angular/common';

@Component({
  selector: 'app-actual',
  imports: [
    NgOptimizedImage
  ],
  templateUrl: './actual.component.html',
  styleUrl: './actual.component.scss'
})
export class ActualComponent {
  letterToSign = input<Lesson>()
}
