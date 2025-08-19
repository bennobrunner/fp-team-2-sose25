import { Routes } from '@angular/router';
import {ModeSelectionComponent} from './pages/mode-selection/mode-selection.component';
import {LessonComponent} from './pages/lesson/lesson.component';
import {DictionaryComponent} from './pages/dictionary/dictionary.component';

export const routes: Routes = [
  {
    path: "learn",
    component: LessonComponent
  },
  {
    path: "dictionary",
    component: DictionaryComponent
  },
  {
    path: "**",
    component: ModeSelectionComponent
  }
];
