export enum NoteType {
  Don = "don",
  Ka = "ka",
  BigDon = "bigDon",
  BigKa = "bigKa",
  Drumroll = "drumroll",
  BigDrumroll = "bigDrumroll",
  Balloon = "balloon",
}

export type CourseNote = { time_ms: number; type: NoteType };

export enum ParseState {
  Neutral = "neutral",
  BranchFinding = "branchfinding",
  EndFinding = "endfinding",
  NoteParsing = "noteparsing",
}