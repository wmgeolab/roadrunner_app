package com.example.roadrunner_geolab;

public class location {
    private double IRIx;
    private double IRIy;
    private double IRIz;
    private int count;

    public location() {
    }

    public location(double IRIx, double IRIy, double IRIz, int count) {
        this.IRIx = IRIx;
        this.IRIy = IRIy;
        this.IRIz = IRIz;
        this.count = count;
    }

    public double getIRIx() {
        return IRIx;
    }

    public void setIRIx(double IRIx) {
        this.IRIx = IRIx;
    }

    public double getIRIy() {
        return IRIy;
    }

    public void setIRIy(double IRIy) {
        this.IRIy = IRIy;
    }

    public double getIRIz() {
        return IRIz;
    }

    public void setIRIz(double IRIz) {
        this.IRIz = IRIz;
    }

    public int getCount() {
        return count;
    }

    public void setCount(int count) {
        this.count = count;
    }
}
