import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { VideoComponent } from './components/video/video.component';
import {LessonComponent} from './pages/lesson/lesson.component';
import {ModeSelectionComponent} from './pages/mode-selection/mode-selection.component';
import {HeaderComponent} from './components/header/header.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, HeaderComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'fp-team2';
}
