"""
UI helper functions for the Bike Flow Explorer app.
"""

def get_nta_index(borough):
    """
    Get default NTA index for a given borough.
    Returns index of a popular NTA in that borough.
    """
    # Default to first NTA alphabetically
    defaults = {
        "Manhattan": 0,  # First Manhattan NTA
        "Brooklyn": 0,
        "Queens": 0,
        "Bronx": 0,
        "Staten Island": 0,
    }
    return defaults.get(borough, 0)


def apply_styles(styler, selected_nta, year_month1, year_month2):
    """
    Apply styling to the ranking dataframe.
    
    Args:
        styler: pandas Styler object
        selected_nta: NTA to highlight
        year_month1: First comparison period
        year_month2: Second comparison period
    
    Returns:
        Styled dataframe
    """
    # 1. Background gradient on Percent Change column
    styler.background_gradient(axis=0, cmap="RdYlGn", subset="Percent Change")
    
    # 2. Format numbers
    styler.format(
        {
            year_month1: "{:,.0f}",
            year_month2: "{:,.0f}",
            "Change": "{:+,.0f}",
            "Percent Change": "{:+.1f}%",
        }
    )
    
    # 3. Highlight the selected NTA row
    def highlight_row(row):
        condition = row["NTA"] == selected_nta
        style = [
            (
                "background-color: #FFFACD; font-weight: bold; color: black"
                if condition and col != "Percent Change"
                else ""
            )
            for col in row.index
        ]
        return style
    
    return styler.apply(lambda _: highlight_row(_), axis=1)


def format_year_month(year_month):
    """Format year_month string (e.g., '202301') to readable format (e.g., 'Jan 2023')."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    year = year_month[:4]
    month_idx = int(year_month[4:]) - 1
    return f"{months[month_idx]} {year}"


def get_comparison_periods(year_months):
    """
    Get sensible default comparison periods.
    Returns (earlier_period, later_period).
    """
    sorted_months = sorted(year_months)
    
    if len(sorted_months) < 2:
        return sorted_months[0], sorted_months[0]
    
    # Default to comparing same month year-over-year if possible
    # Otherwise, compare first and last available periods
    latest = sorted_months[-1]
    latest_year = int(latest[:4])
    latest_month = latest[4:]
    
    # Try to find same month from previous year
    previous_year_same_month = f"{latest_year - 1}{latest_month}"
    
    if previous_year_same_month in sorted_months:
        return previous_year_same_month, latest
    
    # Fall back to comparing recent months
    return sorted_months[-2], sorted_months[-1]
