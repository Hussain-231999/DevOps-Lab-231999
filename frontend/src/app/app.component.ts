import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';

interface PredictionResponse {
  prediction: string;
  delay_probability: number;
  model: string;
}

interface FlightForm {
  MONTH: number;
  DAY: number;
  DAY_OF_WEEK: number;
  AIRLINE: string;
  ORIGIN_AIRPORT: string;
  DESTINATION_AIRPORT: string;
  SCHEDULED_TIME: number;
  DISTANCE: number;
  SCHEDULED_DEPARTURE_HOUR: number;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  private readonly http = inject(HttpClient);
  readonly apiUrl = 'http://localhost:5000';
  readonly form: FlightForm = {
    MONTH: 6,
    DAY: 15,
    DAY_OF_WEEK: 1,
    AIRLINE: 'AA',
    ORIGIN_AIRPORT: 'LAX',
    DESTINATION_AIRPORT: 'JFK',
    SCHEDULED_TIME: 300,
    DISTANCE: 2475,
    SCHEDULED_DEPARTURE_HOUR: 8
  };
  result: PredictionResponse | null = null;
  error = '';
  loading = false;

  predict(): void {
    this.loading = true;
    this.error = '';
    this.result = null;
    this.http.post<PredictionResponse>(`${this.apiUrl}/predict`, this.form).subscribe({
      next: (result) => {
        this.result = result;
        this.loading = false;
      },
      error: (response: { error?: { error?: string } }) => {
        this.error = response.error?.error ?? 'The prediction request failed.';
        this.loading = false;
      }
    });
  }
}
