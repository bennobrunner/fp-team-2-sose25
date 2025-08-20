import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ContinuousRecognitionComponent } from './continuous-recognition.component';

describe('ContinuousRecognitionComponent', () => {
  let component: ContinuousRecognitionComponent;
  let fixture: ComponentFixture<ContinuousRecognitionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ContinuousRecognitionComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ContinuousRecognitionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
