import { Component } from '@angular/core';

@Component({
  selector: 'app-dictionary',
  imports: [],
  templateUrl: './dictionary.component.html',
  styleUrl: './dictionary.component.scss'
})
export class DictionaryComponent {
  allCharacters = [...Array(26).keys()].map(i => String.fromCharCode(65 + i)); // A-Z
}
