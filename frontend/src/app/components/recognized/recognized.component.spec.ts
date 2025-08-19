import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RecognizedComponent } from './recognized.component';

describe('RecognizedComponent', () => {
  let component: RecognizedComponent;
  let fixture: ComponentFixture<RecognizedComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RecognizedComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RecognizedComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
