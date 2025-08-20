import {NormalizedLandmark} from '@mediapipe/hands';

export interface Letter {
  character: LessonLetter;
  landmarks: NormalizedLandmark[][];
}

export type LessonLetter = 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I' | 'J' | 'K' | 'L' | 'M'
  | 'N' | 'O' | 'P' | 'Q' | 'R' | 'S' | 'T' | 'U' | 'V' | 'W' | 'X' | 'Y' | 'Z';

export interface Lesson {
  character: Letter;
}
